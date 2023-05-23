import streamlit as st                                                  #Streamlit importeren
import pandas as pd                                                     #Pandas importeren om data te analyseren en te manipuleren
import numpy as np                                                      #Numpy importeren om matrixcalculaties te kunnen doen
from datetime import datetime                                           #datetime importeren om te kunnen werken met datum en tijd
import datetime                                                         #datetime importeren om te kunnen werken met datum en tijd
import plotly.express as px                                             #Plotly importeren voor het visualiseren van data in interactieve grafieken
import plotly.graph_objs as go                                          #Plotly importeren voor het visualiseren van data in interactieve grafieken
import sqlalchemy                                                       #sqlalchemy importeren om verbinding te kunnen maken met de database
from urllib.parse import quote_plus                                     #quote_plus van urllib.parse importeren om wachtwoorden naar unicode-formaat te kunnen transformeren
from streamlit_extras.dataframe_explorer import dataframe_explorer      #dataframe_explorer van streamlit_extras importeren zodat er een aanpasbaar filter op dataframes gezet kan worden
from streamlit_extras.chart_container import chart_container            #chart_container van streamlit_extras importeren zodat er in 1 visualisatie 3 tabs komen om een grafiek weer te geven, de achterliggende data weer te geven en exportopties weer te geven.
import streamlit_authenticator as stauth                                #streamlit_authenticator importeren voor de login-pagina
import pickle                                                           #Pickle importeren om de gehashte wachtwoorden op te kunnen halen uit het pickle-bestand
from pathlib import Path                                                #Path van pathlib importeren om de bestandslocaties van de pickle-bestanden te kunnen zoeken
import configparser                                                     #configparser importeren om configuratiebestanden in te kunnen lezen

#--- functie om metric-visualisaties een ander uiterlijk te geven ---
def style_metric_cards(
    background_color: str = "#3B3B3A",
    border_size_px: int = 1,
    border_color: str = "#CCC",
    border_radius_px: int = 5,
    border_left_color: str = "#ADCD53",
    box_shadow: bool = True,
):

    box_shadow_str = (
        "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
        if box_shadow
        else "box-shadow: none !important;"
    )
    st.markdown(
        f"""
        <style>
            div[data-testid="metric-container"] {{
                background-color: {background_color};
                border: {border_size_px}px solid {border_color};
                padding: 5% 5% 5% 10%;
                border-radius: {border_radius_px}px;
                border-left: 0.5rem solid {border_left_color} !important;
                {box_shadow_str}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Webpagina configureren ---
st.set_page_config(
    page_title="Warehouse performance | Westerman Multimodal Logistics", 
    page_icon = "🚛",
    layout = 'wide',
    initial_sidebar_state= "collapsed"
)

logo = 'images/westerman-logo-wit.png'
st.sidebar.image(logo, width=300 , use_column_width=False)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#--- Authenticatie ---

names = ["Management Warehouse", "Directie", "Management Engineering"]
usernames = ["wml-warehouse", "wml-directie", 'wml-engineering']

# --- Gehashte wachtwoorden inladen --- 
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    "warehouse_dashboard_cookie", 
                                    "cookietoken_warehouse_dashboard", 
                                    cookie_expiry_days=1) #Cookie plaatsen met een verlooptijd van 1 dag, zodat er 1x per dag ingelogd hoeft te worden als er niet wordt uitgelogd.

name, authentication_status, username = authenticator.login("Login", "main")

# --- Wanneer inloggegevens incorrect ingevoerd zijn ---
if authentication_status == False:
    st.error("Gebruikersnaam of wachtwoord is incorrect.")

# --- Wanneer geen inloggegevens ingevoerd zijn ---
if authentication_status == None:
    st.warning("Voer uw gebruikersnaam en wachtwoord in.")

# --- Wanneer de juiste inloggegevens ingevoerd zijn ---
if authentication_status:
    refresh = st.button("Refresh data", type = "primary") #Refreshknop om alle cache te wissen en applicatie te resetten. Hiermee worden ook de laatste gegevens uit de database opgehaald.
    if refresh:
        st.cache_data.clear()
    #--- Functie om de tijd te onthouden wanneer de refreshknop voor het laatst is ingedrukt.
    @st.cache_data()
    def tijd_refresh():
        nu = datetime.datetime.now()
        current_datetime = nu.strftime("%d-%m-%Y om %H:%M:%S")
        return current_datetime
    @st.cache_data(experimental_allow_widgets=True)
    def tijd_print():
        tijd = tijd_refresh()
        return tijd
    tijd = tijd_print()
    st.text(f"Laatste refresh was op {tijd}")
    
    #--- Connectie maken met de database ---
    config = configparser.ConfigParser()
    config.read('config.ini')

    server_name = '10.80.110.56'
    db_name = 'BC_PROD_087_20.3'
    username = config['database']['username']
    password = config['database']['password']
    #--- Connectie-engine bouwen ---
    engine = sqlalchemy.create_engine(f'mssql+pyodbc://{username}:{quote_plus(password)}@{server_name}/{db_name}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes')

    st.title("📈 Warehouse performance")
    st.sidebar.subheader(f"{name} is nu ingelogd.")  
    authenticator.logout("Log uit", "sidebar")
    #----------------------------------------- LINKER GEDEELTE ---------------------------------------------------
    col1, coltussen, col2 = st.columns((100,5,100))

    with col1:
        st.subheader("Algemene warehouse performance")
        st.markdown("<h6 style='color: white;'>Hier zijn de algemene statistieken<br><br></h6>", unsafe_allow_html=True)
    # --- Functie om data uit de database op te halen. Deze data wordt vervolgens in het cache-geheugen opgeslagen. ---
    @st.cache_data(show_spinner="Data uit database ophalen, even geduld...")
    def load_data():
        artikelposten = pd.read_sql("SELECT [Location No_], [Carrier No_], [Document No_], [User ID], [Modified Date_Time] FROM dbo.[Westerman Multimodal Logistics$WMS Item Ledger Entry$67c29c0f-0df8-4d43-b066-9655f855ec2c] WHERE [Entry Type] = 5 AND [Document No_] LIKE 'U%' AND [Unit of Measure Code] LIKE '%FIE%' AND [Building Code] = 'DG25' AND [Modified Date_Time] >= DATEFROMPARTS(YEAR(GETDATE())-1, 1, 1) ORDER BY [Modified Date_Time] DESC;", engine)
        artikelposten = artikelposten.rename(columns = {
        'Location No_': 'Locatienr.',
        'Carrier No_': 'Dragernr.',
        'Document No_': 'Documentnr.',
        'User ID': 'Gebruikers-id',
        'Modified Date_Time': 'Gewijzigd op'
        })
        artikelposten['Gewijzigd op'] = pd.to_datetime(artikelposten['Gewijzigd op'], utc=True, format="%d-%m-%y %H:%M:%S")
        artikelposten['Gewijzigd op'] = artikelposten['Gewijzigd op'].dt.tz_convert('Europe/Paris').dt.tz_localize(None)

        return artikelposten
    artikelposten_ruw = load_data()
    # --- Transformeren van de data ---
    @st.cache_data(show_spinner= "Data transformeren...")
    def transform_data(artikelposten):    
        artikelposten['Aantal'] = 1
        artikelposten["Datum"] = artikelposten["Gewijzigd op"].dt.date
        artikelposten['Datum'] = pd.to_datetime(artikelposten['Datum'], format="%Y-%m-%d")
        artikelposten.sort_values(['Gebruikers-id', 'Gewijzigd op'], ascending = [True, True], inplace=True)
        artikelposten.loc[artikelposten['Gebruikers-id'] != artikelposten['Gebruikers-id'].shift(), 'Reistijd'] = 0
        artikelposten = artikelposten[["Locatienr.", "Dragernr.", "Documentnr.", "Aantal", "Gebruikers-id", "Datum", "Gewijzigd op"]]
        artikelposten['Vorige locatie'] = artikelposten['Locatienr.'].shift(1)
        artikelposten.loc[artikelposten['Gebruikers-id'] != artikelposten['Gebruikers-id'].shift(), 'Vorige locatie'] = "0"
        artikelposten[['HLL', 'HLC']] = artikelposten['Locatienr.'].str.extract('([a-zA-Z]+)(\d+)', expand=True)
        artikelposten[['VLL', 'VLC']] = artikelposten['Vorige locatie'].str.extract('([a-zA-Z]+)(\d+)', expand=True)
        artikelposten = artikelposten[(artikelposten['VLL'].str.len() <= 2) & ((artikelposten['VLL'].str[:1] == 'Z') | (artikelposten['VLL'].str.len() == 1))]
        artikelposten = artikelposten[(artikelposten['HLL'].str.len() <= 2) & ((artikelposten['HLL'].str[:1] == 'Z') | (artikelposten['HLL'].str.len() == 1))]
        artikelposten = artikelposten[(artikelposten['VLC'].str.len() == 5) & (artikelposten['HLC'].str.len() == 5)]
        artikelposten['Weekdag'] = pd.to_datetime(artikelposten['Datum'])
        artikelposten['Weekdag'] = (artikelposten['Weekdag'].dt.weekday)+1
        return artikelposten
    artikelposten_ruw = transform_data(artikelposten_ruw)
    #--- Medewerker selectie en data hierop filteren ---
    @st.cache_data(show_spinner = "Medewerkers laden...")
    def filter_artikelposten(artikelposten, gebruiker_ids):
        filtered_artikelposten = artikelposten[artikelposten["Gebruikers-id"].isin(gebruiker_ids)]
        return filtered_artikelposten 

    with col1:
        unique_gebruikers = list(artikelposten_ruw["Gebruikers-id"].unique())
        unique_gebruikers.sort(reverse=False)
        unique_gebruikers.insert(0, "Alle medewerkers") # Optie 'Alle medewerkers' toevoegen aan medewerker-selectie
        
        #Selectiewidget voor de medewerker
        gebruiker_ids = st.multiselect(label= 'Selecteer één of meerdere medewerkers', options=unique_gebruikers)

        #--- Herleiden welke gebruikers in welke shift hebben gewerkt, zodat er met overlappende shifts gedeald kan worden door de code ---
        artikelposten_ruw['Gewijzigd op'] = pd.to_datetime(artikelposten_ruw['Gewijzigd op'])
        nacht_picks = artikelposten_ruw[artikelposten_ruw['Gewijzigd op'].dt.time >= pd.to_datetime('23:00:00').time()]
        nacht_picks = nacht_picks.groupby(['Gebruikers-id', 'Datum'])['Aantal'].sum().reset_index()
        nacht_picks['Datum_morgen'] = nacht_picks['Datum'] + pd.DateOffset(days=1)
        nacht_picks['Datum_morgen']= pd.to_datetime(nacht_picks['Datum_morgen'])
        nacht_picks_na12 = artikelposten_ruw[artikelposten_ruw['Gewijzigd op'].dt.time < pd.to_datetime('07:00:00').time()]
        nacht_picks_na12 = nacht_picks_na12.groupby(['Gebruikers-id', 'Datum'])['Aantal'].sum().reset_index()
        nacht_picks_na12['Datum'] = pd.to_datetime(nacht_picks_na12['Datum'])
        nacht_picks_per_medewerker = nacht_picks_na12.merge(nacht_picks, left_on = ['Gebruikers-id','Datum'], right_on = ['Gebruikers-id','Datum_morgen'], how = 'outer')
        nacht_picks_per_medewerker['Aantal_x'] = nacht_picks_per_medewerker['Aantal_x'].fillna(0)
        nacht_picks_per_medewerker['Aantal_y'] = nacht_picks_per_medewerker['Aantal_y'].fillna(0)
        nacht_picks_per_medewerker['Aantal'] = nacht_picks_per_medewerker['Aantal_x']+nacht_picks_per_medewerker['Aantal_y']
        nachtpicks = nacht_picks_per_medewerker.groupby(['Gebruikers-id', 'Datum_x'])['Aantal'].sum().reset_index()
        nachtpicks = nachtpicks.rename(columns={'Datum_x': 'Datum'}) 
        artikelposten_ruw['Gewijzigd op'] = pd.to_datetime(artikelposten_ruw['Gewijzigd op'])
        dagpicks = artikelposten_ruw[((artikelposten_ruw['Gewijzigd op'].dt.time >= pd.to_datetime('07:00:00').time())&(artikelposten_ruw['Gewijzigd op'].dt.time < pd.to_datetime('15:00:00').time()))]
        dagpicks = dagpicks.groupby(['Gebruikers-id', 'Datum'])['Aantal'].sum().reset_index() 
        artikelposten_ruw['Gewijzigd op'] = pd.to_datetime(artikelposten_ruw['Gewijzigd op'])
        avondpicks = artikelposten_ruw[((artikelposten_ruw['Gewijzigd op'].dt.time >= pd.to_datetime('15:00:00').time())&(artikelposten_ruw['Gewijzigd op'].dt.time < pd.to_datetime('23:00:00').time()))]
        avondpicks = avondpicks.groupby(['Gebruikers-id', 'Datum'])['Aantal'].sum().reset_index()      

        # 1. Definieer de tijdsgrenzen voor de verschillende shifts
        dag_start = pd.Timestamp('07:00')
        avond_start = pd.Timestamp('15:00')
        nacht_start = pd.Timestamp('23:00')
        shift_duur = pd.Timedelta(hours=8, minutes=30)

        # 2. Definieer de functie om de shift te bepalen
        def bepaal_shift(row):
            datum_tijd = pd.to_datetime(row['Gewijzigd op'])
            gebruiker = row['Gebruikers-id']
            dag_vd_week = row['Weekdag']
            
            #nacht
            if datum_tijd.time() <= dag_start.time():
                return 3 * (dag_vd_week -1 )
            
            #nacht/dag    
            elif datum_tijd.time() >= dag_start.time() and datum_tijd.time() < (dag_start + shift_duur).time():
                if (gebruiker, datum_tijd.date()) in nachtpicks.index:
                    return 3 * (dag_vd_week -1 )
                else:
                    return 3 * (dag_vd_week - 1) + 1
                
            #avond/dag
            elif datum_tijd.time() >= avond_start.time() and datum_tijd.time() < (avond_start + shift_duur).time():
                if (gebruiker, datum_tijd.date()) in dagpicks.index:
                    return 3 * (dag_vd_week - 1) + 1
                else:
                    return 3 * (dag_vd_week - 1) + 2
                
            #nacht/avond
            elif (datum_tijd.time() >= nacht_start.time()):
                if (gebruiker, datum_tijd.date()) in avondpicks.index:
                    return 3 * (dag_vd_week - 1) + 2
                else:
                    return 3 * (dag_vd_week - 1) + 3

        # 3. Vul de 'Shift'-kolom van het dataframe 'df'
        nachtpicks = nachtpicks.set_index(['Gebruikers-id', 'Datum'])
        dagpicks = dagpicks.set_index(['Gebruikers-id', 'Datum'])
        avondpicks = avondpicks.set_index(['Gebruikers-id', 'Datum'])

        # --- Bepaal per medewerker in welke shift een pick werd uitgevoerd en sla de gegevens in het cache-geheugen op ---
        @st.cache_data(show_spinner= "Gewerkte shifts identificeren...")
        def apply_shift():
            artikelposten_ruw['Shift'] = artikelposten_ruw.apply(bepaal_shift, axis=1)
            return artikelposten_ruw
        
        artikelposten_ruw = apply_shift()   

        col1a, col1b = st.columns((1,1))

        with col1a:
            #Filterwidget zodat de locaties gefilterd worden. Slaat de selectie op in het cachegeheugen. ---
            @st.cache_data(experimental_allow_widgets=True, show_spinner="Selectie-widgets inladen...")
            def picklocatie_selectie():
                picklocaties = st.radio(label="Voor welke locaties wil je de statistieken zien?",
                    key="picklocaties",
                    options=["Stellingen", "Grondlocaties", "Beide"],
                )

                bulk_check = artikelposten_ruw['Locatienr.'].str.contains('BULK')
                mezz_check = artikelposten_ruw['Locatienr.'].str.contains('MEZZ')
                dock_check = artikelposten_ruw['Locatienr.'].str.contains('DOCK')
                outbound_check = artikelposten_ruw['Locatienr.'].str.contains('OUTBOUND')

                if picklocaties == "Stellingen":
                    artikelposten_df = artikelposten_ruw[~((artikelposten_ruw["Locatienr."].str.endswith("01"))|
                                                    (artikelposten_ruw['Vorige locatie'].str.endswith("01"))|
                                                    bulk_check | mezz_check | dock_check | outbound_check)]
                elif picklocaties == "Grondlocaties":
                    artikelposten_df = artikelposten_ruw[(artikelposten_ruw["Locatienr."].str.endswith("01")) &
                                                    (artikelposten_ruw["Vorige locatie"].str.endswith("01"))&
                                                    ~(bulk_check | mezz_check | dock_check | outbound_check)]
                elif picklocaties == 'Beide':
                    artikelposten_df = artikelposten_ruw[~(bulk_check | mezz_check | dock_check | outbound_check)]

                    
                artikelposten_df['Reistijd'] = (artikelposten_df['Gewijzigd op'] - artikelposten_df['Gewijzigd op'].shift()).dt.total_seconds()
                artikelposten_df.loc[artikelposten_df['Gebruikers-id'] != artikelposten_df['Gebruikers-id'].shift(), 'Reistijd'] = 0

                #Gangpadwisselkolom toevoegen
                mask = ((artikelposten_df['HLL'].ne(artikelposten_df['VLL'])) | 
                        (artikelposten_df['Reistijd'].gt(300))) & (artikelposten_df['Reistijd'] <= 900) & ((artikelposten_df['Gebruikers-id'] == artikelposten_df['Gebruikers-id'].shift()) | 
                        ~(artikelposten_df['Gebruikers-id'] != artikelposten_df['Gebruikers-id'].shift()))

                artikelposten_df['Wissel'] = mask.astype(int)

                return artikelposten_df, picklocaties
            
            if len(gebruiker_ids)==0:
                st.empty()
            else:
                artikelposten_df, picklocaties = picklocatie_selectie()
        
        # --- Shift selectietool maken ---
        with col1b:
            shift_dict = {
                0: 'Zondag op maandagnacht',
                1: 'Maandag, dag',
                2: 'Maandagavond',
                3: 'Maandag op dinsdagnacht',
                4: 'Dinsdag, dag',
                5: 'Dinsdagavond',
                6: 'Dinsdag op woensdagnacht',
                7: 'Woensdag, dag',
                8: 'Woensdagavond',
                9: 'Woensdag op donderdagnacht',
                10: 'Donderdag, dag',
                11: 'Donderdagavond',
                12: 'Donderdag op vrijdagnacht',
                13: 'Vrijdag, dag',
                14: 'Vrijdagavond',
                15: 'Vrijdag op zaterdagnacht',
                16: 'Zaterdag, dag',
                17: 'Zaterdagavond'
            }
            if len(gebruiker_ids)==0:
                st.empty()
            else:
                shift_selected = st.multiselect('Selecteer shifts (zonder selectie wordt alles geselecteerd)', list(shift_dict.values()))
        
        #Functie voor de visualisatie 'Aantal picks per gangpad'
        @st.cache_data(show_spinner = "Aantal picks per gangpad laden...")
        def plot_picks_per_gangpad(df):
           
            locatie_letters = ['X','W','V','U','T','S','R','Q','P','O','N','M','L','K','J','I','H','G','F','E','D','C','B','A','ZA','ZB','ZC','ZD','ZE','ZF','ZG','ZH','ZI','ZJ','ZK','ZL','ZM','ZN','ZO','ZP','ZQ','ZR','ZS','ZT','ZU','ZV','ZW','ZX']
            df[['Locatie letters', 'Locatie cijfers']] = df['Locatienr.'].str.extract('([a-zA-Z]+)(\d+)', expand=True)    
            gangpad_aantallen = df.groupby("Locatie letters")['Aantal'].sum().reset_index(name = "Aantallen per gangpad")
            df = df.drop(['Locatie letters', 'Locatie cijfers'], axis = 1)
            trace = go.Bar(
            x= gangpad_aantallen["Locatie letters"],
            y= gangpad_aantallen['Aantallen per gangpad'],
            marker=dict(color='#ADCD53')
                            
            )

            layout = go.Layout(
                paper_bgcolor='#3D3D3B',
                plot_bgcolor='#3D3D3B',
                xaxis=dict(
                    tickfont=dict(color='white'),
                    tickcolor='white',
                    categoryorder='array', 
                    categoryarray= locatie_letters
                ),
                yaxis=dict(
                    title='Aantal picks',
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white')
                ),
                margin=dict(l=50, r=50, t=50, b=50), title= "Aantal picks per gangpad"
            )
            # maak de figuur aan en voeg de trace en layout toe
            fig = go.Figure(data=[trace], layout=layout)
            fig.update_layout(
            plot_bgcolor='#3D3D3B', 
            paper_bgcolor='#3D3D3B', 
            xaxis_tickangle=-50,
            xaxis=dict(title='Datum'), 
            yaxis=dict(title='Aantal picks'),
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            width=600,
            height=400,
            )
            fig.update_traces(hoverlabel_font_size = 32)
            div_style_aantal = '''
                        <style>
                            div.stPlotlyChart {
                                border: 10px solid #3D3D3B;
                                border-radius: 20px;
                            }
                        </style>
                        '''
            st.markdown(div_style_aantal, unsafe_allow_html=True)  
            return fig
        #Functie voor het toewijzen van de hal zodat hierop gefilterd kan worden
        def hal_toewijzen(value):
            if 'Z' in value:
                return 2
            else:
                return 1
        #Data filteren op geselecteerde medewerkers
        if "Alle medewerkers" in gebruiker_ids:
            df = artikelposten_df
        elif len(gebruiker_ids) > 0:
            df = filter_artikelposten(artikelposten_df, gebruiker_ids)
        else:
            df = pd.DataFrame()
   
        #Data vervolgens filteren op geselecteerde shift(s)
        if 'Alle medewerkers' in gebruiker_ids or len(gebruiker_ids)>=1:
            if shift_selected:
                selected_shifts = [key for key, value in shift_dict.items() if value in shift_selected]
                df = df[df['Shift'].isin(selected_shifts)] 

            df = df.assign(Hal=df['Locatienr.'].apply(hal_toewijzen))
            df_print = df[['Gebruikers-id', 'Datum', 'Gewijzigd op', 'Weekdag', 'Shift', 'Documentnr.', 'Dragernr.', 'Locatienr.', 'Vorige locatie', 'Aantal', 'Reistijd', 'Wissel', 'Hal']]
            df_print['Reistijd'] = (df_print['Reistijd'].fillna(0)).astype(int)
            # --- Aanpasbaar filter voor de data ---
            df_teprinten = dataframe_explorer(df_print, case = False)
            
            #--- Selectietool voor de algemene visualisatieselectie ---
            graphselect = st.selectbox("Selecteer een visualisatie:", ["Aantal picks per dag", "Aantal picks per gangpad", "Aantal verwerkte fietsen per medewerker", "Gemiddelde tijd tussen picks per dag"])
            
            #--- Functie om inslagen in te laden toevoegen
            if graphselect == "Aantal verwerkte fietsen per medewerker":
                inslag_knop = st.checkbox("inslagen tonen?", False)
                if inslag_knop == True:
                    # --- inslagen ophalen uit de database. Deze data wordt opgeslagen in het cache-geheugen
                    @st.cache_data(show_spinner = "Inslagen inladen...")
                    def inslagen_query_laden():
                        inslagen_query = pd.read_sql("""SELECT [Location No_] AS 'Locatienr.', [User ID] AS 'Gebruikers-id', 1 AS 'Aantal', CONVERT(date, [Modified Date_Time]) AS 'Datum' FROM dbo.[Westerman Multimodal Logistics$WMS Item Ledger Entry$67c29c0f-0df8-4d43-b066-9655f855ec2c] WHERE [Entry Type] = 6 AND [Document No_] LIKE 'DV%' AND [Unit of Measure Code] LIKE '%FIE%' AND [Building Code] = 'DG25' AND [Modified Date_Time] >= DATEFROMPARTS(YEAR(GETDATE())-1, 1, 1) AND (([Location No_] LIKE '[a-zA-Z][0-9][0-9][0-9][0-9][0-9]') OR ([Location No_] LIKE 'Z[a-zA-Z][0-9][0-9][0-9][0-9][0-9]')) ORDER BY [Modified Date_Time] DESC;""", engine)
                        return inslagen_query
                    inslagen_df = inslagen_query_laden()
                    st.text("Filter hieronder de inslagen.")
                    st.text("! Let op !: De inslagen voor alle locaties en alle shifts worden weergegeven. Hier kan niet op gefilterd worden.")
                    inslagen = dataframe_explorer(inslagen_df, case = False)
                else:
                    inslagen = pd.DataFrame(columns=['Locatienr.', 'Gebruikers-id', 'Aantal', 'Datum'])
            
            #---Functie voor de grafiek voor het aantal picks per medewerker---
            @st.cache_data(show_spinner = "Aantal picks per medewerker laden...")
            def aantal_picks_pm(df_teprinten, inslagen):
                inslagen_aantal = inslagen.groupby(['Gebruikers-id'])['Aantal'].sum().reset_index()
                uitslagen_aantal = df_teprinten.groupby(['Gebruikers-id'])['Aantal'].sum().reset_index()
                trace1 = go.Bar(x=uitslagen_aantal['Gebruikers-id'],
                y=uitslagen_aantal['Aantal'],
                name = 'Orderpicks',
                marker=dict(color='#ADCD53'),
                width = 0.5
                )
                trace2 = go.Bar(x=inslagen_aantal['Gebruikers-id'],
                y=inslagen_aantal['Aantal'],
                name = 'Inslagen',
                marker=dict(color='#19A7CE'),
                width = 0.5
                )

                layout = go.Layout(
                    paper_bgcolor='#3D3D3B',
                    plot_bgcolor='#3D3D3B',
                    xaxis=dict(
                        tickfont=dict(color='white'),
                        tickcolor='white'
                    ),
                    yaxis=dict(
                        title='Aantallen',
                        titlefont=dict(color='white'),
                        tickfont=dict(color='white'),
                    ),
                    margin=dict(l=50, r=50, t=50, b=50), title= "Aantal verwerkte fietsen per medewerker"
                )
                fig = go.Figure(data= [trace1, trace2], layout=layout)
                fig.update_layout(
                plot_bgcolor='#3D3D3B', 
                paper_bgcolor='#3D3D3B', 
                xaxis_tickangle=-50,
                xaxis=dict(title='Medewerker'), 
                yaxis=dict(title='Aantal'),
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False,
                width= 600,
                height=400,
                                    
                )
                fig.update_traces(hoverlabel_font_size = 32)
                div_style_shifts = '''
                            <style>
                                div.stPlotlyChart {
                                    border: 10px solid #3D3D3B;
                                    border-radius: 20px;
                                }
                            </style>
                            '''
                st.markdown(div_style_shifts, unsafe_allow_html=True)
                return fig 
            
            #Dataframe filteren om performance in kaart te brengen: dftijd
            @st.cache_data(show_spinner = "Performance-statistieken laden...")
            def performance_stats(df_teprinten):
                grouped_artikelposten = artikelposten_ruw.groupby('Datum')['Aantal'].sum()
                gaf = grouped_artikelposten[grouped_artikelposten>=180]
                dftijd = df_teprinten[df_teprinten['Datum'].isin(gaf.index)]
                dftijd = dftijd[(dftijd['Reistijd']>0)&(dftijd['Reistijd']<=900)]
                dftijd = dftijd.groupby(['Datum', 'Gebruikers-id']).filter(lambda x: x['Reistijd'].mean() >= 70)
                dftijd = dftijd[~((dftijd['Reistijd']<27)&(dftijd['Wissel']==1))]
                
                aantal_mdwrks = artikelposten_df.groupby('Gebruikers-id').count()['Aantal']
                filtered_ids = aantal_mdwrks[aantal_mdwrks < 500].index
                dftijd = dftijd[~dftijd['Gebruikers-id'].isin(filtered_ids)]

                #Hoogteverschillen dubieuze scans filteren
                dftijd['Gelijk'] = dftijd[['Locatienr.', 'Vorige locatie']].apply(lambda x: x.str[-2:]).eq(dftijd[['Locatienr.', 'Vorige locatie']].apply(lambda x: x.str[-2:])).all(axis=1).astype(int)
                dftijd = dftijd.loc[(dftijd['Reistijd'] >= 10) | (dftijd['Gelijk'] == 0)]
                dftijd = dftijd.drop('Gelijk', axis=1)


                #Stellingsverschillen dubieuze scans filteren
                dftijd[['HLL', 'HLC']] = dftijd['Locatienr.'].str.extract('([a-zA-Z]+)(\d+)', expand=True)
                dftijd[['VLL', 'VLC']] = dftijd['Vorige locatie'].str.extract('([a-zA-Z]+)(\d+)', expand=True)        
                dftijd['HLC'] = pd.to_numeric(dftijd['HLC'], errors='coerce')
                dftijd['VLC'] = pd.to_numeric(dftijd['VLC'], errors='coerce')
                dftijd['HLC_2_3'] = (dftijd['HLC'] // 10 % 100).astype(np.int16)
                dftijd['VLC_2_3'] = (dftijd['VLC'] // 10 % 100).astype(np.int16)
                dftijd['verschil'] = np.abs(dftijd['HLC_2_3'] - dftijd['VLC_2_3'])
                dftijd['stellingen_verschil'] = np.where((dftijd['verschil'] > 4) & (dftijd['Wissel'] == 0), 1, 0)
                dftijd = dftijd.drop(['HLC_2_3', 'VLC_2_3', 'verschil', 'HLC', 'VLC', 'HLL', 'VLL'], axis = 1)
                dftijd = dftijd[~((dftijd['stellingen_verschil'] == 1) & (dftijd['Reistijd'] < 15))]
                dftijd = dftijd.drop('stellingen_verschil', axis = 1)

                #shifttypes toevoegen
                shifttypes = {0: 'Nacht', 1: 'Dag', 2: 'Avond', 3: 'Nacht', 4: 'Dag', 5: 'Avond', 6: 'Nacht', 7: 'Dag', 8: 'Avond', 9: 'Nacht', 10: 'Dag', 11: 'Avond', 12: 'Nacht', 13: 'Dag', 14: 'Avond', 15: 'Nacht', 16: 'Dag', 17: 'Avond', 18: 'Nacht', 19: 'Dag', 20: 'Avond', 21: 'Nacht'}
                dftijd['shifttype'] = dftijd['Shift'].map(shifttypes)
                return dftijd
            
            if df.shape[0]==0:
                dftijd = pd.DataFrame()
            else:
                dftijd = performance_stats(df_teprinten)
            
            #Functie voor de gemiddelde tijd tussen picks
            @st.cache_data(show_spinner = "Gemiddelde tijd tussen picks...")
            def gem_picktijd(df_teprinten):
                df_picktijd = performance_stats(df_teprinten)
                gem_rt_perdag = df_picktijd.groupby(['Datum'])['Reistijd'].mean()
                trendline = np.polyfit(gem_rt_perdag.index.astype(int)/1e9, gem_rt_perdag, 1)
                y_trendline = np.polyval(trendline, gem_rt_perdag.index.astype(int)/1e9)
                trace = go.Scatter(x=gem_rt_perdag.index,
                y=gem_rt_perdag,
                mode = 'lines+markers',
                name = "Tijd",
                marker=dict(color='#ADCD53', size = 6),
                line = dict(color='#ADCD53', width = 3)
                )
                trendline_trace = go.Scatter(x=gem_rt_perdag.index,
                y=y_trendline,
                mode='lines',
                name = "Trend",
                line = dict(color='#ED2B2A', width = 3)
                )
                layout = go.Layout(
                    paper_bgcolor='#3D3D3B',
                    plot_bgcolor='#3D3D3B',
                    xaxis=dict(
                        tickfont=dict(color='white'),
                        tickcolor='white'
                    ),
                    yaxis=dict(
                        title='Gemiddelde tijd tussen picks',
                        titlefont=dict(color='white'),
                        tickfont=dict(color='white'),
                    ),
                    margin=dict(l=50, r=50, t=50, b=50), title= "Gemiddelde tijd tussen picks"
                )
                fig = go.Figure(data=[trace, trendline_trace], layout=layout)
                fig.update_layout(
                plot_bgcolor='#3D3D3B', 
                paper_bgcolor='#3D3D3B', 
                xaxis_tickangle=-50,
                xaxis=dict(title='Datum'), 
                yaxis=dict(title='Gemiddelde tijd tussen picks'),
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False,
                width= 600,
                height=400,
                                    
                )
                fig.update_traces(hoverlabel_font_size = 32)
                div_style_shifts = '''
                            <style>
                                div.stPlotlyChart {
                                    border: 10px solid #3D3D3B;
                                    border-radius: 20px;
                                }
                            </style>
                            '''
                st.markdown(div_style_shifts, unsafe_allow_html=True)
                return fig  

            #---Chart container toevoegen waarin alle visualisatiefuncties weergegeven kunnen worden, samen met de ruwe data en de exportmogelijkheden ---
            with chart_container(df_teprinten):
                if graphselect == "Aantal picks per dag":
                    picks_per_dag = df_teprinten.groupby('Datum')['Aantal'].sum().reset_index()
                    picks_per_dag["Datum"] = pd.to_datetime(picks_per_dag['Datum'], format = '%Y-%m-%d')

                    trace = go.Bar(
                    x= picks_per_dag["Datum"],
                    y= picks_per_dag['Aantal'],
                    marker=dict(color='#ADCD53')
                                        
                    )
                    layout = go.Layout(
                        paper_bgcolor='#3D3D3B',
                        plot_bgcolor='#3D3D3B',
                        xaxis=dict(
                            tickfont=dict(color='white'),
                            tickcolor='white'
                        ),
                        yaxis=dict(
                            title='Aantal picks',
                            titlefont=dict(color='white'),
                            tickfont=dict(color='white')
                        ),
                        margin=dict(l=50, r=50, t=50, b=50), title= "Aantal picks per dag",
                        showlegend = False
                    )
                    # maak de figuur aan en voeg de trace en layout toe
                    fig = go.Figure(data=[trace], layout=layout)
                    fig.update_layout(
                    plot_bgcolor='#3D3D3B', 
                    paper_bgcolor='#3D3D3B', 
                    xaxis_tickangle=-50,
                    xaxis=dict(title='Datum'), 
                    yaxis=dict(title='Aantal picks'),
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False,
                    width=600,
                    height=400,
                    )
                    fig.update_traces(hoverlabel_font_size = 32)
                    div_style_aantal = '''
                                <style>
                                    div.stPlotlyChart {
                                        border: 10px solid #3D3D3B;
                                        border-radius: 20px;
                                    }
                                </style>
                                '''
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(div_style_aantal, unsafe_allow_html=True)
                elif graphselect=="Aantal picks per gangpad":
                    st.plotly_chart(plot_picks_per_gangpad(df_teprinten), use_container_width=True)
                elif graphselect == "Aantal verwerkte fietsen per medewerker":
                    st.plotly_chart(aantal_picks_pm(df_teprinten, inslagen), use_container_width = True)
                elif graphselect == "Gemiddelde tijd tussen picks per dag":
                    if dftijd.shape[0]==0:
                        st.write(f"{', '.join(str(id) for id in gebruiker_ids)} heeft niet genoeg picks gedaan om inzicht te kunnen geven in persoonlijke performance-statistieken.")
                    else:
                        st.plotly_chart(gem_picktijd(df_teprinten), use_container_width = True)
                  
    #----------------------------------------- RECHTER GEDEELTE ---------------------------------------------------
    
    # --- Verdere styling van de metric-visuals ---
    st.markdown(
        """
        <style>
        [data-testid="stMetricDelta"] svg {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with col2:
        if len(gebruiker_ids)>0:
            st.subheader("Performance per orderpicker")
            
            if "Alle medewerkers" in gebruiker_ids:
                #---Metric visualisaties maken voor alle medewerkers---
                st.markdown("<h6 style='color: white; margin-bottom: 18px;'>Hier zijn de statistieken voor alle medewerkers met performance-inzichten:</h6>", unsafe_allow_html=True)
                norm = st.number_input("Wat is de normtijd voor de gemiddelde tijd tussen picks (in seconden)?", 0, 250, 125, 1)
                col2a, col2b, col2c = st.columns((1,1,1))
                gem_rt = dftijd['Reistijd'].mean()
                gem_ppg = (df_teprinten['Aantal'].sum())/(df_teprinten['Wissel'].sum())
                tot_p = df_teprinten['Aantal'].sum()
                tot_p1 = df_teprinten[df_teprinten['Hal']==1]['Aantal'].sum()
                tot_p2 = df_teprinten[df_teprinten['Hal']==2]['Aantal'].sum()
                d_value1 = str(round(gem_rt-norm, 2))+ " sec. van de normtijd"
                d_value3 = "Hal 1: " + str(tot_p1) + " - Hal 2: " + str(tot_p2)
                
                col2a.metric(label = "Gemiddelde tijd tussen picks", value = str(round(gem_rt, 2))+" (sec)", delta = d_value1, delta_color="inverse", )
                col2b.metric(label = "Gemiddeld aantal picks per gangpad", value = str(round(gem_ppg, 3))+" fietsen")
                col2c.metric(label = "Aantal picks in periode", value = str(tot_p) + " fietsen", delta = d_value3, delta_color = "off")
                style_metric_cards()
            elif len(gebruiker_ids)==1:
                if dftijd.shape[0]==0:
                    st.text(f"{', '.join(str(id) for id in gebruiker_ids)} heeft niet genoeg picks gedaan om inzicht te kunnen geven in persoonlijke performance-statistieken.")
                else:
                    #---Metric visualisaties maken voor 1 medewerker ---
                    st.markdown("<h6 style='color: white; margin-bottom: 18px;'>Hier zijn de statistieken voor de geselecteerde medewerker:</h6>", unsafe_allow_html=True)
                    norm = st.number_input("Wat is de normtijd voor de gemiddelde tijd tussen picks (in seconden)?", 0, 250, 125, 1)
                    col2a, col2b, col2c = st.columns((1,1,1))
                    gem_rt = dftijd['Reistijd'].mean()
                    gem_ppg = (df_teprinten['Aantal'].sum())/(df_teprinten['Wissel'].sum())
                    tot_p = df_teprinten['Aantal'].sum()
                    tot_p1 = df_teprinten[df_teprinten['Hal']==1]['Aantal'].sum()
                    tot_p2 = df_teprinten[df_teprinten['Hal']==2]['Aantal'].sum()
                    d_value1 = str(round(gem_rt-norm, 2))+ " sec. van de normtijd"
                    d_value3 = "Hal 1: " + str(tot_p1) + " - Hal 2: " + str(tot_p2)

                    col2a.metric(label = "Gemiddelde tijd tussen picks", value = str(round(gem_rt, 2))+" (sec)", delta = d_value1, delta_color="inverse")
                    col2b.metric(label = "Gemiddeld aantal picks per gangpad", value = str(round(gem_ppg, 3))+" fietsen")
                    col2c.metric(label = "Aantal picks in periode", value = str(tot_p) + " fietsen", delta = d_value3, delta_color = "off")
                    style_metric_cards()
            elif len(gebruiker_ids)>1:
                if dftijd.shape[0]==0:
                    st.text("De geselecteerde medewerkers hebben niet genoeg picks gedaan om inzicht te kunnen geven in persoonlijke performance-statistieken.")
                else:
                    #--- Metric visualisaties maken voor meer dan 1 medewerker ---
                    st.markdown("<h6 style='color: white;'>Hier zijn de statistieken voor de geselecteerde medewerkers:<br>! Let op ! : Over sommige geselecteerde medewerkers worden mogelijk geen persoonlijke statistieken weergegeven. Je kunt dit checken door één medewerker tegelijk te selecteren.</h6>", unsafe_allow_html=True)
                    norm = st.number_input("Wat is de normtijd voor de gemiddelde tijd tussen picks (in seconden)?", 0, 250, 125, 1)
                    col2a, col2b, col2c = st.columns((1,1,1))
                    gem_rt = dftijd['Reistijd'].mean()
                    gem_ppg = (df_teprinten['Aantal'].sum())/(df_teprinten['Wissel'].sum())
                    tot_p = df_teprinten['Aantal'].sum()
                    tot_p1 = df_teprinten[df_teprinten['Hal']==1]['Aantal'].sum()
                    tot_p2 = df_teprinten[df_teprinten['Hal']==2]['Aantal'].sum()
                    d_value1 = str(round(gem_rt-norm, 2))+ " sec. van de normtijd"
                    d_value3 = "Hal 1: " + str(tot_p1) + " - Hal 2: " + str(tot_p2)
                    
                    col2a.metric(label = "Gemiddelde tijd tussen picks", value = str(round(gem_rt, 2))+" (sec)", delta = d_value1, delta_color="inverse")
                    col2b.metric(label = "Gemiddeld aantal picks per gangpad", value = str(round(gem_ppg, 3))+" fietsen")
                    col2c.metric(label = "Aantal picks in periode", value = str(tot_p) + " fietsen", delta = d_value3, delta_color = "off")
                    style_metric_cards()
        # --- Optie tot uitsluiten van medewerkers toevoegen ---
        if len(gebruiker_ids)>0:
            if dftijd.shape[0]>0:
                if "Alle medewerkers" in gebruiker_ids:   
                    uitsluiting_medewerkers = st.multiselect("Selecteer eventuele medewerkers die je wilt uitsluiten.", dftijd['Gebruikers-id'].tolist())
                    dftijd = dftijd[~dftijd['Gebruikers-id'].isin(uitsluiting_medewerkers)]          

                col2a2, col2a2tussen, col2b2 = st.columns((100, 1, 100))
                with col2a2:
                    # --- Plotly visualisatie voor de gemiddelde tijd tussen picks per shifttype
                    gem_rt_st = dftijd.groupby(['shifttype'])['Reistijd'].mean()
                    trace = go.Bar(x=gem_rt_st,
                    y=gem_rt_st.index,
                    orientation = 'h',
                    marker=dict(color='#ADCD53'),
                    width = 0.5
                    )
                    layout = go.Layout(
                        paper_bgcolor='#3D3D3B',
                        plot_bgcolor='#3D3D3B',
                        xaxis=dict(
                            tickfont=dict(color='white'),
                            tickcolor='white'
                        ),
                        yaxis=dict(
                            title='Gemiddelde tijd tussen picks',
                            titlefont=dict(color='white'),
                            tickfont=dict(color='white'),
                            categoryorder='array', 
                            categoryarray= ['Nacht', 'Avond', 'Dag']
                        ),
                        margin=dict(l=50, r=50, t=50, b=50), title= "Gemiddelde tijd tussen picks"
                    )
                    fig = go.Figure(data=[trace], layout=layout)
                    fig.update_layout(
                    plot_bgcolor='#3D3D3B', 
                    paper_bgcolor='#3D3D3B', 
                    xaxis_tickangle=-50,
                    xaxis=dict(title='Seconden'), 
                    yaxis=dict(title='Shift'),
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False,
                    width= 600,
                    height=400,
                                        
                    )
                    fig.update_traces(hoverlabel_font_size = 32)
                    div_style_shifts = '''
                                <style>
                                    div.stPlotlyChart {
                                        border: 10px solid #3D3D3B;
                                        border-radius: 20px;
                                    }
                                </style>
                                '''
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(div_style_shifts, unsafe_allow_html=True)
                with col2b2:
                    if len(gebruiker_ids) == 0:
                        st.empty()
                    elif "Alle medewerkers" in gebruiker_ids or len(gebruiker_ids)>1:  
                        # ---Plotly visualisatie voor de gemiddelde tijd tussen picks per medewerker met extra info per medewerker ---
                        gem_rt_pp = dftijd.groupby('Gebruikers-id')['Reistijd'].mean().sort_values(ascending=False)
                        picks_pp = df_teprinten.groupby('Gebruikers-id')['Aantal'].sum().loc[gem_rt_pp.index]

                        trace = go.Bar(x=gem_rt_pp,
                                    y=gem_rt_pp.index,
                                    customdata=picks_pp,
                                    hovertemplate='<b>%{y}</b><br><br>Gemiddelde tijd: %{x:.2f}<br>Aantal picks: %{customdata}',
                                    orientation='h',
                                    name = 'info',
                                    marker=dict(color='#ADCD53'),
                                    width=0.5
                        )
                        layout = go.Layout(
                            paper_bgcolor='#3D3D3B',
                            plot_bgcolor='#3D3D3B',
                            xaxis=dict(
                                tickfont=dict(color='white'),
                                tickcolor='white'
                            ),
                            yaxis=dict(
                                title='Gemiddelde tijd tussen picks',
                                titlefont=dict(color='white'),
                                tickfont=dict(color='white')

                            ),
                            margin=dict(l=50, r=50, t=50, b=50), title= "Gemiddelde tijd tussen picks"
                        )
                        fig = go.Figure(data=[trace], layout=layout)
                        fig.update_layout(
                        plot_bgcolor='#3D3D3B', 
                        paper_bgcolor='#3D3D3B', 
                        xaxis_tickangle=-50,
                        xaxis=dict(title='Seconden'), 
                        yaxis=dict(title='Medewerker'),
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                        width= 600,
                        height=400,
                                            
                        )
                        fig.update_traces(hoverlabel_font_size = 32)
                        div_style_shifts = '''
                                    <style>
                                        div.stPlotlyChart {
                                            border: 10px solid #3D3D3B;
                                            border-radius: 20px;
                                        }
                                    </style>
                                    '''
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(div_style_shifts, unsafe_allow_html=True)

                    else: 
                        #--- Plotly-visualisatie voor 1 medewerker waarbij de momenten dat een medewerker pickt inzichtelijk is. ---
                        user_picks = df_teprinten
                        user_picks['pick'] = range(1, len(user_picks) + 1)
                        aantalpicks_medewerker = user_picks['Aantal'].sum()
                        fig = px.scatter(user_picks, x='Gewijzigd op', y='pick', hover_data=["Locatienr."], hover_name= "Dragernr.",
                                    title=f'{str(gebruiker_ids[0])} deed {aantalpicks_medewerker} picks (gemiddeld {round(gem_rt, 2)} sec. per pick)', color_discrete_sequence=['#ADCD53'])
                        fig.update_layout(
                            plot_bgcolor='#3D3D3B', 
                            paper_bgcolor='#3D3D3B', 
                            xaxis_tickangle=-50,
                            xaxis=dict(title='Tijd'), 
                            yaxis=dict(title='Aantal picks'),
                            margin=dict(l=0, r=0, t=30, b=0),
                            showlegend=False,
                            width=600,
                            height=400,
                        )
                        div_style = '''
                        <style>
                            div.stPlotlyChart {
                                border: 10px solid #3D3D3B;
                                border-radius: 20px;
                            }
                        </style>
                        '''
                        fig.update_traces(hoverlabel_font_size = 32)
                        fig.update_xaxes(showgrid=False)
                        st.plotly_chart(fig, use_container_width=True, sizing_mode = 'scale')
                        st.markdown(div_style, unsafe_allow_html=True)
# --- EINDE VAN HET SCRIPT VOOR WAREHOUSE PERFORMANCE ---
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        