#--- Importeren van benodigde modules/libraries ---

import streamlit as st                                              #Streamlit importeren
import pandas as pd                                                 #Pandas importeren om data te analyseren en te manipuleren
import numpy as np                                                  #Numpy importeren om matrixcalculaties te kunnen doen
import datetime                                                     #datetime importeren om te kunnen werken met datum en tijd
from scipy.stats import zscore                                      #zscore van scipy.stats importeren om statistische uitbaters te kunnen verwijderen
from scipy import stats                                             #stats van scipy importeren om statistische bewerkingen op data te kunnen uitvoeren

import sqlalchemy                                                   #sqlalchemy importeren om verbinding te kunnen maken met de database
from urllib.parse import quote_plus                                 #quote_plus van urllib.parse importeren om wachtwoorden naar unicode-formaat te kunnen transformeren

import streamlit_authenticator as stauth                            #streamlit_authenticator importeren voor de login-pagina
import pickle                                                       #Pickle importeren om de gehashte wachtwoorden op te kunnen halen uit het pickle-bestand
from pathlib import Path                                            #Path van pathlib importeren om de bestandslocaties van de pickle-bestanden te kunnen zoeken
import configparser                                                 #configparser importeren om configuratiebestanden in te kunnen lezen
from streamlit_option_menu import option_menu                       #option_menu van streamlit_option_menu importeren om de short-term/long-term widget te kunnen maken (geen standaard widget)

from streamlit_extras.dataframe_explorer import dataframe_explorer  #dataframe_explorer van streamlit_extras importeren zodat er een aanpasbaar filter op dataframes gezet kan worden

#--- Webpagina configureren ---
st.set_page_config(
    page_title="Orderpicking | Westerman Multimodal Logistics", 
    page_icon = "🚛",
    layout = 'wide',
    initial_sidebar_state= "collapsed"
)
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

#--- stylen van de webpagina ---
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

#--- gehashte wachtwoorden inladen ---
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)

#--- inlogvenster maken ---
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    "warehouse_dashboard_cookie", 
                                    "cookietoken_warehouse_dashboard", 
                                    cookie_expiry_days=1) #Cookie plaatsen met een verlooptijd van 1 dag, zodat er 1x per dag ingelogd hoeft te worden als er niet wordt uitgelogd.

name, authentication_status, username = authenticator.login("Login", "main")

#--- Wanneer inloggegevens onjuist zijn ingevoerd ---
if authentication_status == False:
    st.error("Gebruikersnaam of wachtwoord is incorrect.")

#--- Wanneer er geen inloggegevens zijn ingevoerd ---
if authentication_status == None:
    st.warning("Voer uw gebruikersnaam en wachtwoord in.")

# --- Wanneer de inloggegevens correct zijn ingevoerd ---
if authentication_status:
    col1refresh, col2navbar = st.columns((3,6))
    with col1refresh:
        refresh = st.button("Refresh data", type = "primary") #Refreshknop maken zodat alle gecachete gegevens verwijderd worden en de website volledig wordt herladen

        if refresh:
            st.cache_data.clear()
        # --- Tijdfunctie voor de refreshknop zodat bekend is wanneer de laatste refresh was ---    
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
    #--- Navigatiebar voor short-term/long-term ---
    with col2navbar:
        selected=option_menu(
            menu_title=None,
            options=["short-term planning", "long-term planning"],
            icons=["calendar-check", "chevron-double-right"],
            default_index=0,
            orientation="horizontal"
        )
    
    #--- Inladen van gegevens en koppeling maken met de database ---
    config = configparser.ConfigParser()
    config.read('config.ini')

    server_name = '10.80.110.56'
    db_name = 'BC_PROD_087_20.3'
    username = config['database']['username']
    password = config['database']['password']

    engine = sqlalchemy.create_engine(f'mssql+pyodbc://{username}:{quote_plus(password)}@{server_name}/{db_name}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes')

    st.title("🚲 Personeel & Materieel")
    st.sidebar.subheader(f"{name} is nu ingelogd.")  
    authenticator.logout("Log uit", "sidebar")
    #----------------------------------------- LINKER KOLOM ---------------------------------------------------
    col1, coltussen, col2 = st.columns((100,5, 100))
    # --- Data ophalen uit database en cachen (opslaan) in cache-geheugen ---
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
    artikelposten = load_data()
    # --- Data transformeren, filteren en nieuwe kolommen toevoegen ---
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
    artikelposten = transform_data(artikelposten)



    with col1:
        #--- periode bepalen waarvoor performance-statistieken van medewerkers worden bepaald ---
        coldatum1, coldatum2 = st.columns((3,1))
        if 'datum_vanaf' not in st.session_state:
            datumwaarde = pd.to_datetime(artikelposten['Datum'].min())
        else:
            datumwaarde = st.session_state['datum_vanaf']
        datum_vanaf = coldatum1.date_input("Performance-statistieken bepalen vanaf (klik op berekenen om datum op te slaan):",min_value = pd.to_datetime(artikelposten['Datum']).min(), value = datumwaarde, max_value = datetime.datetime.now(), on_change = None, key = 'datum_vanaf', help = "De performance-statistieken vanaf de geselecteerde datum tot vandaag worden berekend. Klik op berekenen na het selecteren van de datum. Het opnieuw laden van de performance-statistieken kan enige tijd duren. Houd rekening met de reset van de planningsparameters.")
        coldatum2.markdown("<h6 style='color: white; margin-bottom: 19px;'></h6>", unsafe_allow_html=True)
        if coldatum2.button("Berekenen", use_container_width = True, type = "primary"):
            st.cache_data.clear()
        
        artikelposten = artikelposten[artikelposten['Datum']>=pd.to_datetime(st.session_state['datum_vanaf'])]
        unique_gebruikers = list(artikelposten["Gebruikers-id"].unique())
        unique_gebruikers.sort(reverse=False)
        unique_gebruikers.insert(0, "Alle medewerkers")
        gebruiker_ids = unique_gebruikers

        #--- Herleiden welke gebruikers in welke shift hebben gewerkt, zodat er met overlappende shifts gedeald kan worden door de code ---
        artikelposten['Gewijzigd op'] = pd.to_datetime(artikelposten['Gewijzigd op'])
        nacht_picks = artikelposten[artikelposten['Gewijzigd op'].dt.time >= pd.to_datetime('23:00:00').time()]
        nacht_picks = nacht_picks.groupby(['Gebruikers-id', 'Datum'])['Aantal'].sum().reset_index()
        nacht_picks['Datum_morgen'] = nacht_picks['Datum'] + pd.DateOffset(days=1)
        nacht_picks['Datum_morgen']= pd.to_datetime(nacht_picks['Datum_morgen'])
        nacht_picks_na12 = artikelposten[artikelposten['Gewijzigd op'].dt.time < pd.to_datetime('07:00:00').time()]
        nacht_picks_na12 = nacht_picks_na12.groupby(['Gebruikers-id', 'Datum'])['Aantal'].sum().reset_index()
        nacht_picks_na12['Datum'] = pd.to_datetime(nacht_picks_na12['Datum'])
        nacht_picks_per_medewerker = nacht_picks_na12.merge(nacht_picks, left_on = ['Gebruikers-id','Datum'], right_on = ['Gebruikers-id','Datum_morgen'], how = 'outer')
        nacht_picks_per_medewerker['Aantal_x'] = nacht_picks_per_medewerker['Aantal_x'].fillna(0)
        nacht_picks_per_medewerker['Aantal_y'] = nacht_picks_per_medewerker['Aantal_y'].fillna(0)
        nacht_picks_per_medewerker['Aantal'] = nacht_picks_per_medewerker['Aantal_x']+nacht_picks_per_medewerker['Aantal_y']
        nachtpicks = nacht_picks_per_medewerker.groupby(['Gebruikers-id', 'Datum_x'])['Aantal'].sum().reset_index()
        nachtpicks = nachtpicks.rename(columns={'Datum_x': 'Datum'}) 
        artikelposten['Gewijzigd op'] = pd.to_datetime(artikelposten['Gewijzigd op'])
        dagpicks = artikelposten[((artikelposten['Gewijzigd op'].dt.time >= pd.to_datetime('07:00:00').time())&(artikelposten['Gewijzigd op'].dt.time < pd.to_datetime('15:00:00').time()))]
        dagpicks = dagpicks.groupby(['Gebruikers-id', 'Datum'])['Aantal'].sum().reset_index() 
        artikelposten['Gewijzigd op'] = pd.to_datetime(artikelposten['Gewijzigd op'])
        avondpicks = artikelposten[((artikelposten['Gewijzigd op'].dt.time >= pd.to_datetime('15:00:00').time())&(artikelposten['Gewijzigd op'].dt.time < pd.to_datetime('23:00:00').time()))]
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
            artikelposten['Shift'] = artikelposten.apply(bepaal_shift, axis=1)
            return artikelposten
        
        artikelposten = apply_shift()   

        col1a, col1b = st.columns((1,1))

        with col1a:
            #--- Filter man-uptruck/grondlocaties met de selectietool st.radio, sla keuze op in cache-geheugen ---
            @st.cache_data(experimental_allow_widgets=True, show_spinner="Nog even geduld...")
            def picklocatie_selectie():
                picklocaties = st.radio(label="Voor welk materieel wordt er een planning gemaakt?",
                    key="picklocaties",
                    options=["Man-up truck", "Elektropallettruck"],
                )

                bulk_check = artikelposten['Locatienr.'].str.contains('BULK')
                mezz_check = artikelposten['Locatienr.'].str.contains('MEZZ')
                dock_check = artikelposten['Locatienr.'].str.contains('DOCK')
                outbound_check = artikelposten['Locatienr.'].str.contains('OUTBOUND')

                if picklocaties == "Man-up truck":
                    artikelposten_df = artikelposten[~((artikelposten["Locatienr."].str.endswith("01"))|
                                                    (artikelposten['Vorige locatie'].str.endswith("01"))|
                                                    bulk_check | mezz_check | dock_check | outbound_check)]
                elif picklocaties == "Elektropallettruck":
                    artikelposten_df = artikelposten[(artikelposten["Locatienr."].str.endswith("01")) &
                                                    (artikelposten["Vorige locatie"].str.endswith("01"))&
                                                    ~(bulk_check | mezz_check | dock_check | outbound_check)]

                artikelposten_df['Reistijd'] = (artikelposten_df['Gewijzigd op'] - artikelposten_df['Gewijzigd op'].shift()).dt.total_seconds()
                #--- Nieuwe kolom toevoegen om aan te geven of er sprake is van een gangpadwissel ---
                mask1 = artikelposten_df['HLL'].ne(artikelposten_df['VLL'])
                mask2 = artikelposten_df['Gebruikers-id'].ne(artikelposten_df['Gebruikers-id'].shift())
                mask3 = artikelposten_df['Reistijd'].gt(300)
                mask = mask1 | mask2 | mask3
                artikelposten_df['Gangpadwisselgroep'] = mask.cumsum()
                artikelposten_df['Wissel'] = artikelposten_df['Gangpadwisselgroep'].ne(artikelposten_df['Gangpadwisselgroep'].shift()).astype(int)
                artikelposten_df = artikelposten_df.drop(['Gangpadwisselgroep', 'HLL', 'VLL'], axis = 1)


                return artikelposten_df, picklocaties
            
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
            shift_selected = st.selectbox('Voor welke shift wordt er een planning gemaakt?', list(shift_dict.values()))
            if shift_selected:
                selected_shifts = [key for key, value in shift_dict.items() if value in shift_selected]
                df = artikelposten_df[artikelposten_df['Shift'].isin(selected_shifts)]
                df_alle = artikelposten_df
        # Historische data met een shift geselecteerd
        df = df[['Datum', 'Gewijzigd op', 'Gebruikers-id', 'Locatienr.', 'Aantal', 'Vorige locatie', 'Reistijd', 'Wissel', 'Weekdag', 'Shift']]
        # Historische data voor alle shifts
        df_alle = df_alle[['Datum', 'Gewijzigd op', 'Gebruikers-id', 'Locatienr.', 'Aantal', 'Vorige locatie', 'Reistijd', 'Wissel', 'Weekdag', 'Shift']]
        
        # --- Performance-statistieken bepalen door filters toe te voegen volgens fase 2 van de infographic (zie onderzoeksverslag, bijlage 7). Performance-statistieken opslaan in cache-geheugen--- 
        @st.cache_data(show_spinner = "Performance-statistieken laden...")
        def performance_stats(df):
            grouped_artikelposten = artikelposten.groupby('Datum')['Aantal'].sum()
            gaf = grouped_artikelposten[grouped_artikelposten>=180]
            dftijd = df[df['Datum'].isin(gaf.index)]
            dftijd = dftijd[(dftijd['Reistijd']>0)&(dftijd['Reistijd']<=900)]
            dftijd = dftijd.groupby(['Datum', 'Gebruikers-id']).filter(lambda x: x['Reistijd'].mean() >= 70)
            dftijd = dftijd[~((dftijd['Reistijd']<27)&(dftijd['Wissel']==1))]
            

            aantal_mdwrks = artikelposten.groupby('Gebruikers-id').count()['Aantal']
            filtered_ids = aantal_mdwrks[aantal_mdwrks < 500].index
            dftijd = dftijd[~dftijd['Gebruikers-id'].isin(filtered_ids)]

            #Hoogteverschillen dubieuze scans filteren
            dftijd['Gelijk'] = dftijd[['Locatienr.', 'Vorige locatie']].apply(lambda x: x.str[-2:]).eq(dftijd[['Locatienr.', 'Vorige locatie']].apply(lambda x: x.str[-2:])).all(axis=1).astype(int)
            dftijd = dftijd.loc[(dftijd['Reistijd'] >= 15) | (dftijd['Gelijk'] == 0)]
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
            dftijd['Shifttype'] = dftijd['Shift'].map(shifttypes)
            dftijd = dftijd.rename(columns = {'Aantal' : 'stat_Aantal'})
            return dftijd
        dftijd = performance_stats(df) #Met shiftselectie
        dftijd_alle = performance_stats(df_alle) #Voor alle shifts
        
        #Waar met performance wordt gewerkt wordt de data uit dftijd gehanteerd. Als er met aantallen uit df (en niet dftijd) wordt gewerkt, wordt dit duidelijk aangegeven: Aantal komt uit df, stat_Aantal komt uit dftijd.

        # --- Prestatie-overzicht maken en opslaan in cache-geheugen ---
        @st.cache_data(show_spinner = "Prestatie-overzicht genereren...")
        def nieuw_df(dftijd, df):
            dftijd_medewerker_datum = dftijd.groupby(['Gebruikers-id', 'Datum', 'Shift', 'Shifttype'])[['Reistijd', 'stat_Aantal']].agg({'Reistijd': 'mean', 'stat_Aantal': 'sum'}).reset_index()
            dftijd_medewerker = dftijd_medewerker_datum.groupby(['Gebruikers-id','Shift', 'Shifttype'])[['Reistijd', 'stat_Aantal']].agg({'Reistijd': ['mean', 'std'], 'stat_Aantal': 'sum'}).reset_index()
            dftijd_medewerker.columns = ['Gebruikers-id', 'Shift', 'Shifttype', 'Reistijd', 'Reistijd_std', 'stat_Aantal']

            medewerkers = unique_gebruikers
            shifts = df['Shift'].unique()
            data = pd.DataFrame({'Gebruikers-id':medewerkers})
            data['Shift'] = data['Gebruikers-id'].apply(lambda x: shifts)
            data = data.explode('Shift')
            data = data.sort_values(['Gebruikers-id', 'Shift'], ascending = [True, True])

            dag_dict = {
                1: 'Maandag',
                2: 'Maandag',
                3: 'Dinsdag',
                4: 'Dinsdag',
                5: 'Dinsdag',
                6: 'Woensdag',
                7: 'Woensdag',
                8: 'Woensdag',
                9: 'Donderdag',
                10: 'Donderdag',
                11: 'Donderdag',
                12: 'Vrijdag',
                13: 'Vrijdag',
                14: 'Vrijdag',
                15: 'Zaterdag',
                16: 'Zaterdag',
                17: 'Zaterdag',
                18: 'Zondag',
                19: 'Zondag',
                20: 'Zondag',
                0: 'Maandag'
            }

            def shift_to_dag(shift):
                return dag_dict.get(shift)

            shifttypes = {0: 'Nacht', 1: 'Dag', 2: 'Avond', 3: 'Nacht', 4: 'Dag', 5: 'Avond', 6: 'Nacht', 7: 'Dag', 8: 'Avond', 9: 'Nacht', 10: 'Dag', 11: 'Avond', 12: 'Nacht', 13: 'Dag', 14: 'Avond', 15: 'Nacht', 16: 'Dag', 17: 'Avond', 18: 'Nacht', 19: 'Dag', 20: 'Avond', 21: 'Nacht'}
            
            dftijd_medewerker = pd.merge(data, dftijd_medewerker, how = 'left', on=['Gebruikers-id', 'Shift'])
            dftijd_medewerker['Shifttype'] = dftijd_medewerker['Shift'].map(shifttypes)
            dftijd_medewerker['stat_Aantal'] = dftijd_medewerker['stat_Aantal'].fillna(0).astype(int)
            dftijd_medewerker['Dag'] = dftijd_medewerker['Shift'].apply(shift_to_dag)
            return dftijd_medewerker
        dftijd_medewerker = nieuw_df(dftijd, df) #prestatie-overzicht met shiftselectie
        stats_gevuld_alle = nieuw_df(dftijd_alle, df_alle) #prestatieoverzicht zonder shiftselectie

        # --- Dataframe ontwikkelen waarin performance-statistieken staan voor shifts waarin medewerkers niet hebben gewerkt zodat deze ook in andere shifts ingepland kunnen worden. ---
        @st.cache_data(show_spinner = "Statistieken verzamelen...")
        def stats_collect(stats_gevuld, stats_gevuld_alle):
            stats_gevuld.loc[stats_gevuld['stat_Aantal']<100, 'Reistijd'] = None
            stats_gevuld.loc[stats_gevuld['stat_Aantal']<100, 'Reistijd_std'] = None

            means_dag = stats_gevuld_alle.groupby(['Gebruikers-id', 'Dag'])['Reistijd'].mean().reset_index()
            means_shift = stats_gevuld_alle.groupby(['Gebruikers-id', 'Shifttype'])['Reistijd'].mean().reset_index()
            means_overall = stats_gevuld_alle.groupby(['Gebruikers-id'])['Reistijd'].mean().reset_index()
            means_std = stats_gevuld_alle.groupby(['Gebruikers-id'])['Reistijd_std'].mean().reset_index()

            stats_gevuld = pd.merge(stats_gevuld, means_dag, how = 'left', on = ['Gebruikers-id', 'Dag'], suffixes = ("", "_dag"))
            stats_gevuld = pd.merge(stats_gevuld, means_shift, how = 'left', on = ['Gebruikers-id', 'Shifttype'], suffixes = ("", "_shift"))
            stats_gevuld = pd.merge(stats_gevuld, means_overall, how = 'left', on = ['Gebruikers-id'], suffixes = ("", "_gemiddeld"))
            stats_gevuld = pd.merge(stats_gevuld, means_std, how = 'left', on = ['Gebruikers-id'], suffixes = ("", "_gemiddeld"))

            stats_gevuld['rt_sqrt'] = np.sqrt(stats_gevuld['Reistijd_dag']*stats_gevuld['Reistijd_shift'])
            stats_gevuld['Reistijd'] = np.where(stats_gevuld['Reistijd'].isna(), stats_gevuld['rt_sqrt'], stats_gevuld['Reistijd'])
            stats_gevuld['Reistijd'] = np.where(stats_gevuld['Reistijd'].isna(), stats_gevuld['Reistijd_dag'], stats_gevuld['Reistijd'])
            stats_gevuld['Reistijd'] = np.where(stats_gevuld['Reistijd'].isna(), stats_gevuld['Reistijd_shift'], stats_gevuld['Reistijd'])
            stats_gevuld['Reistijd'] = np.where(stats_gevuld['Reistijd'].isna(), stats_gevuld['Reistijd_gemiddeld'], stats_gevuld['Reistijd'])
            stats_gevuld['Reistijd_std'] = np.where(stats_gevuld['Reistijd_std'].isna(), stats_gevuld['Reistijd_std_gemiddeld'], stats_gevuld['Reistijd_std'])
            stats_gevuld = stats_gevuld.drop(['Reistijd_dag', 'Reistijd_shift', 'rt_sqrt', 'Reistijd_gemiddeld', 'Reistijd_std_gemiddeld'], axis = 1)
            
            nan_pickers_rt = stats_gevuld[stats_gevuld['Reistijd'].isna()]['Gebruikers-id'].unique()
            nan_pickers_std = stats_gevuld[stats_gevuld['Reistijd_std'].isna()]['Gebruikers-id'].unique()
            stats_gevuld = stats_gevuld[~(stats_gevuld['Gebruikers-id'].isin(nan_pickers_rt))]
            stats_gevuld = stats_gevuld[~(stats_gevuld['Gebruikers-id'].isin(nan_pickers_std))]  
            
            stats_gevuld.sort_values(['Gebruikers-id', 'Shift'], ascending = [True, True])   
            stats_gevuld = stats_gevuld[['Gebruikers-id', 'Dag', 'Shifttype', 'Shift', 'stat_Aantal', 'Reistijd', 'Reistijd_std']]
            stats_gevuld['std_gem'] = stats_gevuld.groupby(['Gebruikers-id'])['Reistijd_std'].transform('mean')
            avg_reistijd = stats_gevuld.groupby(['Shift', 'Gebruikers-id'])['Reistijd'].mean().reset_index()
            avg_reistijd = avg_reistijd.sort_values(['Shift', 'Reistijd'])
            if avg_reistijd['Reistijd'].nunique() == 1:
                avg_reistijd['Categorie'] = 'A'
            else:
                avg_reistijd['Categorie'] = avg_reistijd.groupby('Shift')['Reistijd'].apply(lambda x: pd.qcut(x, 6, labels=['A', 'B', 'C', 'D', 'E', 'F']))
            avg_reistijd = avg_reistijd.drop('Reistijd', axis = 1)
            df_stats = pd.merge(stats_gevuld, avg_reistijd, how = 'left', on = ['Shift', 'Gebruikers-id'])
            df_stats.columns = ['Medewerker', 'Dag', 'Shift', 'Shiftnr.', 'stat_Aantal', 'Reistijd', 'Reistijd_std', 'Reistijd_gemstd', 'Categorie']
            z_scores_stds = stats.zscore(df_stats['Reistijd_std'])
            outliers_std = (z_scores_stds > 3) | (z_scores_stds < -3)
            mean_reistijd_std = df_stats['Reistijd_std'].mean()
            df_stats.loc[outliers_std, 'Reistijd_std'] = mean_reistijd_std
            df_stats['Reistijd'] = round(df_stats['Reistijd'], 1)
            df_stats['Reistijd_std'] = round(df_stats['Reistijd_std'], 1)
            df_stats['min_fietsen_per_uur'] = round((3600/(df_stats['Reistijd']+df_stats['Reistijd_std'])),1)
            df_stats['gem_fietsen_per_uur'] = round((3600/df_stats['Reistijd']), 1)
            df_stats['max_fietsen_per_uur'] = round((3600/(df_stats['Reistijd']-df_stats['Reistijd_std'])), 1)
            df_stats = df_stats[['Categorie', 'Medewerker','Dag', 'Shift', 'Reistijd', 'Reistijd_std', 'stat_Aantal', 'min_fietsen_per_uur', 'gem_fietsen_per_uur', 'max_fietsen_per_uur']]
            return df_stats
        
        df_stats = stats_collect(dftijd_medewerker, stats_gevuld_alle) #Dataframe met geselecteerde shift
        df_stats = df_stats[['Categorie', 'Medewerker', 'Reistijd', 'Reistijd_std', 'stat_Aantal', 'min_fietsen_per_uur', 'gem_fietsen_per_uur', 'max_fietsen_per_uur']]  
        stats_alle = stats_collect(stats_gevuld_alle, stats_gevuld_alle) #Dataframe met alle shifts


        # --- Uitwerking voor short-term planning, informatiedataframe per medewerker ---
        if selected == 'short-term planning':
            with st.expander("Informatie per medewerker"):
                alle_shifts = st.checkbox("Informatie voor alle shifts tonen", False)
                if alle_shifts == True:
                    stats_alle_df = dataframe_explorer(stats_alle, case = False)
                    st.dataframe(stats_alle_df, use_container_width=True)
                elif alle_shifts == False:
                    df_stats_df = dataframe_explorer(df_stats, case = False)
                    st.dataframe(df_stats_df, use_container_width=True)               
        # --- Uitwerking voor long-term planning, informatiedataframe per categorie medewerker --- 
        elif selected == 'long-term planning':
            with st.expander("Informatie per categorie medewerker"):
                stats_stat = df_stats
                stats_stat = stats_stat[(zscore(stats_stat['Reistijd_std']) <= 3) & (zscore(stats_stat['Reistijd_std']) >= -3)]
                stats_stat = stats_stat[stats_stat['stat_Aantal']>=100]
                stats_stat = stats_stat[(zscore(stats_stat['min_fietsen_per_uur']) <= 3) & (zscore(stats_stat['min_fietsen_per_uur']) >= -3)]
                stats_stat = stats_stat[(zscore(stats_stat['gem_fietsen_per_uur']) <= 3) & (zscore(stats_stat['gem_fietsen_per_uur']) >= -3)]
                stats_stat = stats_stat[(zscore(stats_stat['max_fietsen_per_uur']) <= 3) & (zscore(stats_stat['max_fietsen_per_uur']) >= -3)]
                stats_cat = stats_stat.groupby('Categorie')[["Reistijd", 
                                                        "Reistijd_std", 
                                                        "stat_Aantal", 
                                                        "min_fietsen_per_uur", 
                                                        "gem_fietsen_per_uur", 
                                                        "max_fietsen_per_uur"]].agg({'Reistijd': 'mean',
                                                                                     'Reistijd_std': 'mean',
                                                                                     'stat_Aantal': 'sum',
                                                                                     'min_fietsen_per_uur': 'mean',
                                                                                     'gem_fietsen_per_uur': 'mean',
                                                                                     'max_fietsen_per_uur': 'mean'}).reset_index()
                stats_cat['Reistijd'] = round(stats_cat['Reistijd'], 1)
                stats_cat['Reistijd_std'] = round(stats_cat['Reistijd_std'], 1)
                stats_cat['min_fietsen_per_uur'] = round(stats_cat['min_fietsen_per_uur'], 1)
                stats_cat['gem_fietsen_per_uur'] = round(stats_cat['gem_fietsen_per_uur'], 1)
                stats_cat['max_fietsen_per_uur'] = round(stats_cat['max_fietsen_per_uur'], 1)
                st.dataframe(stats_cat, use_container_width=True)
        
        # --- Planningsparameters invoerscherm ---
        with st.expander("Voer hier de planningsparameters in:"):
            @st.cache_data(experimental_allow_widgets=True)
            def planning_parameters():
                with st.form("Planningsparameters"):
                    aantal_picks = st.number_input(f"Voer het aantal picks in dat door de {picklocaties.lower()}-chauffeurs in de {shift_selected.lower()}-shift moet worden gedaan:", 0, 50000, 500, 5)
                    norm = st.number_input(f"Voer de normtijd in voor het gemiddeld aantal seconden per pick voor een {picklocaties.lower()}-chauffeur.", 0, 250, 125, 1)
                    if "Zaterdag" in shift_selected:
                        loontoeslag = 46.75
                    elif "zaterdag" in shift_selected:
                        loontoeslag = 46.75
                    else:
                        loontoeslag = 34.30
                    loonkosten = st.number_input(f"Wat is het gemiddelde uurtarief van een {picklocaties.lower()}-chauffeur?", 0.00, 250.00, loontoeslag, 0.01)
                    overige = st.number_input(f"Wat zijn de totale overige kosten voor de {shift_selected.lower()}-shift?", 0, 10000, 295, 1)
                    st.form_submit_button("Invoer opslaan", type = "primary")
                return aantal_picks, norm, loonkosten, overige
            aantal_picks, norm, loonkosten, overige = planning_parameters()

        st.subheader(f"Voer de personeelsplanning voor de {picklocaties.lower()}-chauffeurs voor de {shift_selected.lower()}-shift hieronder in:")
        # --- Short-term planning invoerscherm
        if selected == "short-term planning":
            # --- 3 gemiddelde medewerkers aan medewerkerselectie toevoegen zodat een medewerker als 'gemiddelde medewerker' kan worden ingepland als deze nog te weinig gepickt heeft om performance-statistieken voor te kunnen genereren.---
            nieuwe_regels = [[None, f'Gemiddelde medewerker {i+1}', df_stats['Reistijd'].mean(), df_stats['Reistijd_std'].mean(), 0,
                              3600 / (df_stats['Reistijd'].mean() + df_stats['Reistijd_std'].mean()),
                              3600 / df_stats['Reistijd'].mean(),
                              3600 / (df_stats['Reistijd'].mean() - df_stats['Reistijd_std'].mean())]
                              for i in range(3)
                              ]

            # Maak een nieuwe dataframe met de herhaalde regels
            nieuwe_dataframe = pd.DataFrame(nieuwe_regels, columns=['Categorie', 
                                                                    'Medewerker', 
                                                                    'Reistijd', 
                                                                    'Reistijd_std',
                                                                    'stat_Aantal', 
                                                                    'min_fietsen_per_uur', 
                                                                    'gem_fietsen_per_uur', 
                                                                    'max_fietsen_per_uur'])

            # Voeg de nieuwe dataframe toe aan de bestaande dataframe df_stats
            df_stats = df_stats.append(nieuwe_dataframe, ignore_index=True)
            
            planning_kaal = df_stats[['Medewerker']]

            with st.form("Planning invoeren"):
                with st.expander("Uitleg"):
                    st.write("Hieronder vind je de invoervelden om de planning te maken.") 
                    st.write("Vul bij 'Aantal uren' in hoeveel uren een medewerker werkt. Voorbeeld: Een medewerker werkt 8 en een half uur. Vul dan 8.5 in.")
                    st.write("Vul bij '% Orderpicking' in hoeveel tijd een medewerker moet spenderen aan orderpicken. Als een medewerker de helft van de arbeidstijd moet orderpicken, vul dan 50% in.")
                
                medewerker_selectie = st.multiselect("Selecteer de medewerkers die moeten worden ingepland:", list(planning_kaal["Medewerker"].unique()))

                mcol1, mcol2 = st.columns((5,5))
                m1 = mcol1.number_input("Aantal uren per medewerker", 0.0, 16.0, 0.0, 0.25, key = "m1", help = "Geef hier aan hoeveel uren er gemiddeld per medewerker gewerkt wordt in de geselecteerde shift (incl. pauze).")
                m2 = mcol2.slider("% Orderpicking", 0, 100, 50, 5, help = "Selecteer hier hoeveel arbeidstijd aan orderpicken wordt besteed. Aantal inslagen vs. aantal uitslagen vs. tijd voor overig werk is hier een goede graadmeter. Pauzes en overige onderbrekingen kosten ongeveer 10% van de totale arbeidstijd. Wanneer een medewerker 50% van zijn tijd aan het picken is en 50% inslaat, hanteer dan een totale arbeidstijd van 90%, wat neerkomt op 45% orderpicking.", key = "m2")
                
                
                ColorMinMax = st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
                    background: rgb(1 1 1 / 0%); } </style>''', unsafe_allow_html = True)


                Slider_Cursor = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
                    background-color: rgb(14, 38, 74); box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html = True)

                    
                Slider_Number = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                                { color: rgb(255, 255, 255); } </style>''', unsafe_allow_html = True)
                    

                col = f''' <style> div.stSlider > div[data-baseweb = "slider"] > div > div {{
                    background: linear-gradient(to right, rgb(1, 183, 158) 0%, 
                                                rgb(1, 183, 158) {m2}%, 
                                                rgba(151, 166, 195, 0.25) {m2}%, 
                                                rgba(151, 166, 195, 0.25) 100%); }} </style>'''

                ColorSlider = st.markdown(col, unsafe_allow_html = True) 


                st.form_submit_button("Planning opslaan", type = "primary")

        # --- Long-term planning invoerscherm ---
        if selected == "long-term planning":
            # --- Kwaliteit personeel verandering in de toekomst aangeven, dit doorvoeren op de performance-statistieken ---
            kwaliteit_personeel = st.number_input("Verandering overall performance van personeel (%)", min_value = -20, max_value = 20, value = 0, step = 1)
            if kwaliteit_personeel != 0:
                stats_cat['Reistijd'] = stats_cat['Reistijd']*(1-((kwaliteit_personeel)/100))
                stats_cat['Reistijd_std'] = stats_cat['Reistijd_std']*(1-((kwaliteit_personeel)/100))
                stats_cat['min_fietsen_per_uur'] = stats_cat['min_fietsen_per_uur']*(1+((kwaliteit_personeel)/100))
                stats_cat['gem_fietsen_per_uur'] = stats_cat['gem_fietsen_per_uur']*(1+((kwaliteit_personeel)/100))
                stats_cat['max_fietsen_per_uur'] = stats_cat['max_fietsen_per_uur']*(1+((kwaliteit_personeel)/100))
                if kwaliteit_personeel>0:
                    waarde_is = f"De overall performance van personeel zal met {(abs(kwaliteit_personeel))} procent stijgen ten opzichte van nu."
                else:
                    waarde_is = f"De overall performance van personeel zal met {(abs(kwaliteit_personeel))} procent dalen ten opzichte van nu."
            else:
                waarde_is = "De overall performance van personeel zal gelijk blijven ten opzichte van nu."

            # --- Uitlegscherm ---
            with st.form("Planning invoeren"):
                with st.expander("Uitleg"):
                    st.write(f"Hierboven kun je aangeven hoe je verwacht dat de algemene prestaties van de {picklocaties.lower()}-chauffeurs zich zullen ontwikkelen.")
                    st.write("Hieronder vind je het invoervenster om de planning te maken.") 
                    st.write("Vul bij 'Aantal medewerkers' in hoeveel medewerkers uit deze categorie ingepland worden.")
                    st.write("Vul bij 'Aantal uren' in hoeveel uren een medewerker uit deze categorie (gemiddeld) werkt.")
                    st.write("Vul bij '% Orderpicking' in hoeveel tijd de medewerkers moeten spenderen aan orderpicken. Hiervoor kun je de slider gebruiken.")
                st.text(waarde_is)
               
                # --- parameters verder invullen per categorie medewerker ---
                aantal1, aantal2 = st.columns((1,1))
                Ainput = aantal1.number_input("Selecteer het aantal A-medewerkers", 0, 20, 0, 1, key= "Ainput")
                Binput = aantal1.number_input("Selecteer het aantal B-medewerkers", 0, 20, 0, 1, key= "Binput")
                Cinput = aantal1.number_input("Selecteer het aantal C-medewerkers", 0, 20, 0, 1, key= "Cinput")
                Dinput = aantal2.number_input("Selecteer het aantal D-medewerkers", 0, 20, 0, 1, key= "Dinput")
                Einput = aantal2.number_input("Selecteer het aantal E-medewerkers", 0, 20, 0, 1, key= "Einput")
                Finput = aantal2.number_input("Selecteer het aantal F-medewerkers", 0, 20, 0, 1, key= "Finput")
                
                st.write("\n \n")
                
                uren1, uren2 = st.columns((1, 2))
                orderpicking = st.slider("% Orderpicking", 0, 100, 50, 5, help = "Selecteer hier hoeveel arbeidstijd aan orderpicken wordt besteed. Aantal inslagen vs. aantal uitslagen vs. tijd voor overig werk is hier een goede graadmeter. Pauzes en overige onderbrekingen kosten ongeveer 10% van de totale arbeidstijd. Wanneer een medewerker 50% van zijn tijd aan het picken is en 50% inslaat, hanteer dan een totale arbeidstijd van 90%, wat neerkomt op 45% orderpicking.", key = "orderpicking")
                aantal_uren = uren1.number_input("Aantal uren per medewerker", 0.0, 16.0, 0.0, 0.25, key = "Aantal_uren", help = "Geef hier aan hoeveel uren er gemiddeld per medewerker gewerkt wordt in de geselecteerde shift (incl. pauze).")
                ColorMinMax = st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
                    background: rgb(1 1 1 / 0%); } </style>''', unsafe_allow_html = True)


                Slider_Cursor = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
                    background-color: rgb(14, 38, 74); box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html = True)

                    
                Slider_Number = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                                { color: rgb(255, 255, 255); } </style>''', unsafe_allow_html = True)
                    

                col = f''' <style> div.stSlider > div[data-baseweb = "slider"] > div > div {{
                    background: linear-gradient(to right, rgb(1, 183, 158) 0%, 
                                                rgb(1, 183, 158) {orderpicking}%, 
                                                rgba(151, 166, 195, 0.25) {orderpicking}%, 
                                                rgba(151, 166, 195, 0.25) 100%); }} </style>'''

                ColorSlider = st.markdown(col, unsafe_allow_html = True)   

                st.form_submit_button("Planning opslaan", type = "primary")

    #----------------------------------------- RECHTER KOLOM ---------------------------------------------------

    with col2:
        st.subheader("Verwachting op basis van planning & historische data")
        col21, col22, col23 = st.columns((1,1,1))
        # --- Short-term planning visualisaties weergeven ---
        if selected == 'short-term planning':
            if medewerker_selectie == []:
                st.empty()
            else:
                #Duur, tijden en norm
                duur_picks_short = aantal_picks/((df_stats[df_stats['Medewerker'].isin(medewerker_selectie)]['gem_fietsen_per_uur'].sum())*(m2/100))
                min_duur_picks_short = aantal_picks/((df_stats[df_stats['Medewerker'].isin(medewerker_selectie)]['min_fietsen_per_uur'].sum())*(m2/100))
                std_uren = min_duur_picks_short - duur_picks_short

                norm_picks_short = aantal_picks/(len(medewerker_selectie)*(3600/norm)*(m2/100))

                if ((m1/duur_picks_short*100)-100)>0:
                    delta_duur_short_check = "% boven benodigde capaciteit"
                else:
                    delta_duur_short_check = "% onder benodigde capaciteit"

                delta_duur_short = str(round((m1/duur_picks_short*100)-100, 2))+delta_duur_short_check
                delta_norm_short = str(round((norm_picks_short - duur_picks_short), 2)) + " uur verschil met verwachting"

                col21.metric("Picks naar verwachting uitgevoerd in:", str(round(duur_picks_short,2)) + " uur", delta = delta_duur_short)
                st.write("")
                col22.metric("Picks volgens normtijd uitgevoerd in:", str(round(norm_picks_short, 2)) + " uur", delta = delta_norm_short)
                st.write(f"Met de huidige planning duurt het naar verwachting tussen de **{round(duur_picks_short - std_uren, 1)}** uur en **{round(duur_picks_short + std_uren, 1)}** uur om de orderpicks uit te voeren.")
                
                #Kosten
                kostenvar_short = len(medewerker_selectie)*m1*loonkosten
                factor_op_short = (m2/100)
                kostenoverig_short = overige
                kosten_totaal_short = kostenvar_short*factor_op_short + kostenoverig_short
                delta_kosten_short = "Totaal: € " + str(round(kosten_totaal_short, 2))
                col23.metric("Kostenindicatie per pick", "€ " + str(round(kosten_totaal_short/aantal_picks, 2)), delta = delta_kosten_short, delta_color = "off")
                
                max_cap_planning = round((m1/duur_picks_short)*aantal_picks)
                max_cap_norm = round((m1/norm_picks_short)*aantal_picks)

                st.write(f"Uitgaande van de performance-statistieken kan de huidige bezetting in een gemiddelde situatie maximaal **{max_cap_planning}** fietsen picken.")
                st.write(f"Uitgaande van de norm zou de huidige bezetting maximaal **{max_cap_norm}** fietsen moeten kunnen picken.")
                style_metric_cards()
                
                een_dag_geleden = pd.to_datetime(datetime.date.today() - datetime.timedelta(days = 1))

                df_actueel = df_alle[(df_alle['Datum']>= een_dag_geleden)]
                if shift_selected in shift_dict.values():
                    sleutel = next(key for key, value in shift_dict.items() if value == shift_selected)
                    df_actueel = df_actueel[df_actueel['Shift']==sleutel]
                
                #--- Dataframe met informatie over ingeplande medewerkers ---
                with st.expander("Meer informatie over de ingeplande medewerkers:"):
                    st.write(df_stats[df_stats['Medewerker'].isin(medewerker_selectie)])
                
                # --- Dataframe met activiteit van medewerkers in de geselecteerde shift ---
                if df_actueel.shape[0] == 0:
                    st.empty()
                else:
                    with st.expander("Activiteit van medewerkers in de geselecteerde shift"):
                        act1col, act2col = st.columns((1,4))
                        with act1col:
                            pickers_aan_het_werk = df_actueel.groupby('Gebruikers-id')['Aantal'].sum().reset_index().sort_values('Aantal', ascending = False)
                            st.dataframe(pickers_aan_het_werk, use_container_width = True)
                        with act2col:
                            pickinfo = df_actueel[['Gewijzigd op', 'Gebruikers-id', 'Locatienr.', 'Vorige locatie', 'Reistijd']]
                            pickinfo.columns = ['Tijdstip', 'Medewerker', 'Locatie', 'Vorige locatie', 'Tijd tussen picks']
                            st.dataframe(pickinfo, use_container_width=True)

                # --- Progressie bijhouden in de geselecteerde shift, verwachting voor de afronding van het aantal ingeplande picks, etc. ---
                if df_actueel.shape[0] == 0:
                    st.empty()
                else:
                    actual_in_shift = df_actueel['Aantal'].sum()
                    
                    if "nacht" in shift_selected:
                        now = datetime.datetime.now().time()
                        start_time = datetime.datetime.strptime('23:00:00', '%H:%M:%S').time()
                        end_time = datetime.datetime.strptime('23:59:59', '%H:%M:%S').time()

                        if now >= start_time and now <= end_time:
                            uren_sinds_begin = (now.hour - start_time.hour) + (now.minute - start_time.minute) / 60 + (now.second - start_time.second) / 3600
                        else:
                            uren_sinds_begin = (now.hour + 1) + (now.minute) / 60 + (now.second) / 3600

                    elif "avond" in shift_selected:
                        start_time = datetime.datetime.strptime('15:00:00', '%H:%M:%S').time()
                        now = datetime.datetime.now().time()
                        uren_sinds_begin = (now.hour - start_time.hour) + (now.minute - start_time.minute) / 60 + (now.second - start_time.second) / 3600

                    else:
                        start_time = datetime.datetime.strptime('07:00:00', '%H:%M:%S').time()
                        now = datetime.datetime.now().time()
                        uren_sinds_begin = (now.hour - start_time.hour) + (now.minute - start_time.minute) / 60 + (now.second - start_time.second) / 3600

                    predicted_in_shift = round(((uren_sinds_begin/duur_picks_short)*aantal_picks))
                    normtijd_in_shift = round(((uren_sinds_begin/norm_picks_short)*aantal_picks))
                    
                    if uren_sinds_begin > duur_picks_short:
                        if uren_sinds_begin/m1 > 1 or actual_in_shift > aantal_picks:
                            st.write("Deze shift is al afgerond.")
                        else:
                            st.write(f"Volgens de planning moeten de **{aantal_picks}** orderpicks klaar zijn.")
                            st.progress(uren_sinds_begin/m1, "Tijd in huidige shift")
                            st.progress(actual_in_shift/aantal_picks, "Werkelijke voortgang")
                    else:
                        if uren_sinds_begin > m1 or actual_in_shift > aantal_picks or predicted_in_shift > aantal_picks:
                            st.write("Deze shift is al afgerond")
                        else:
                            st.progress(uren_sinds_begin/m1, f"Tijd in huidige shift: {round(uren_sinds_begin, 2)} uur/{m1} uur.")
                            st.progress(actual_in_shift/aantal_picks, f"Werkelijke voortgang: {actual_in_shift}/{aantal_picks} orderpicks.")
                            st.progress(predicted_in_shift/aantal_picks, f"Voortgang volgens planning: {predicted_in_shift}/{aantal_picks} orderpicks.")
                            if normtijd_in_shift > aantal_picks:
                                st.write("Volgens de normtijd zouden alle orderpicks nu uitgevoerd moeten zijn.")
                            else:
                                st.progress(normtijd_in_shift/aantal_picks, f"Doelstelling om norm te behalen: {normtijd_in_shift}/{aantal_picks} orderpicks.")
                            if uren_sinds_begin > 0.1:
                                tempo = (uren_sinds_begin/actual_in_shift)*60
                                te_doen = aantal_picks - actual_in_shift
                                te_gaan = tempo * te_doen
                                klaar = datetime.datetime.now() + datetime.timedelta(minutes = int(te_gaan))
                                echt_klaar = klaar.strftime("%d-%m-%Y om %H:%M:%S uur")
                                st.write(f"Met het huidige tempo zijn de overige **{te_doen}** picks klaar over **{round(te_gaan, 1)}** minuten (**{round(te_gaan/60, 2)}** uur). Dat is op {echt_klaar}.")
        # --- Long-term visualisaties weergeven ---            
        elif selected == 'long-term planning': 
            medewerkers_selectie_long = (Ainput + Binput + Cinput + Dinput + Einput + Finput)
            if medewerkers_selectie_long == 0:
                st.empty()
            else:
                #Duur, tijden en norm
                capaciteitA = Ainput*(stats_cat[stats_cat['Categorie'] == "A"]['gem_fietsen_per_uur']).sum()
                capaciteitB = Binput*(stats_cat[stats_cat['Categorie'] == "B"]['gem_fietsen_per_uur']).sum()
                capaciteitC = Cinput*(stats_cat[stats_cat['Categorie'] == "C"]['gem_fietsen_per_uur']).sum()
                capaciteitD = Dinput*(stats_cat[stats_cat['Categorie'] == "D"]['gem_fietsen_per_uur']).sum()
                capaciteitE = Einput*(stats_cat[stats_cat['Categorie'] == "E"]['gem_fietsen_per_uur']).sum()
                capaciteitF = Finput*(stats_cat[stats_cat['Categorie'] == "F"]['gem_fietsen_per_uur']).sum()
                capaciteit_long = capaciteitA + capaciteitB + capaciteitC + capaciteitD + capaciteitE + capaciteitF
                duur_picks_long = aantal_picks/(capaciteit_long*(orderpicking/100))

                if ((aantal_uren/duur_picks_long*100)-100)>0:
                    delta_duur_long_check = "% boven benodigde capaciteit"
                else:
                    delta_duur_long_check = "% onder benodigde capaciteit"

                delta_duur_long = str(round((aantal_uren/duur_picks_long*100)-100, 2))+delta_duur_long_check

                col21.metric("Orderpicks naar verwachting uitgevoerd in:", str(round(duur_picks_long,2)) + " uur", delta = delta_duur_long)

                norm_picks_long = aantal_picks/((medewerkers_selectie_long)*(3600/norm)*(orderpicking/100))
                delta_norm_long = str(round((norm_picks_long - duur_picks_long), 2)) + " uur verschil met verwachting"
                col22.metric("Picks volgens normtijd uitgevoerd in:", str(round(norm_picks_long, 2)) + " uur", delta = delta_norm_long)

                #Kosten
                kostenvar_long = (Ainput + Binput + Cinput + Dinput + Einput + Finput)*aantal_uren*loonkosten
                factor_op_long = orderpicking/100
                kostenoverig_long = overige
                kosten_totaal_long = kostenvar_long*factor_op_long + kostenoverig_long
                delta_kosten_long = "Totaal: € " + str(round(kosten_totaal_long, 2))
                col23.metric("Kostenindicatie per pick", "€ " + str(round(kosten_totaal_long/aantal_picks, 2)), delta = delta_kosten_long, delta_color = "off")

                style_metric_cards()
        col24, col25 = st.columns((1,1))
        #---Benodigd materieel weergeven volgens planning---
        if selected == 'long-term planning':
            medewerkers_selectie_long = (Ainput + Binput + Cinput + Dinput + Einput + Finput)
            if medewerkers_selectie_long == 0:
                st.empty()
            else:
                max_cap_planning = round((aantal_uren/duur_picks_long)*aantal_picks)
                max_cap_norm = round((aantal_uren/norm_picks_long)*aantal_picks)
                col24.metric("Maximale capaciteit volgens performance-statistieken", str(max_cap_planning) + " orderpicks")
                col25.metric("Maximale capaciteit volgens norm", str(max_cap_norm) + " orderpicks")
                benodigd_materieel = str(medewerkers_selectie_long) + f" {picklocaties.lower()}(s)"
                st.metric("Benodigd materieel", benodigd_materieel) 
# --- EINDE VAN HET SCRIPT VOOR PERSONEEL & MATERIEEL ---                

                
        
        

        



                












        

