import streamlit as st                      #Streamlit importeren
import pickle                               #Pickle importeren om de gehashte wachtwoorden op te kunnen halen
from pathlib import Path                    #Path van pathlib importeren zodat het script naar het bestandspad van de gehaste passwords kan zoeken
import streamlit_authenticator as stauth    #Streamlit Authenticator importeren zodat er een login-scherm gemaakt kan worden
from PIL import Image                       #Image van PIL importeren zodat afbeeldingen in de app weergegeven kunnen worden


# --- Webpagina configureren ---
st.set_page_config(
    page_title="Westerman Multimodal Logistics",
    page_icon = "🚛", 
    layout = 'wide',
    initial_sidebar_state= "collapsed"
)

# --- Styling van de webpagina ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
logo = 'https://www.westermanlogistics.com/wp-content/themes/provenwebconcepts/assets/img/logo/westerman-logo-wit.svg'
st.sidebar.image(logo, width=300 , use_column_width=False)

#--- Authenticatie ---

names = ["Management Warehouse", "Directie", "Management Engineering"]
usernames = ["wml-warehouse", "wml-directie", 'wml-engineering']

# --- Inladen van de gehashte wachtwoorden ---
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    "warehouse_dashboard_cookie", 
                                    "cookietoken_warehouse_dashboard", 
                                    cookie_expiry_days=1)

name, authentication_status, username = authenticator.login("Log in om deze app te kunnen gebruiken", "main")


# --- Wanneer inloggegevens incorrect zijn ---
if authentication_status == False:
    st.error("Gebruikersnaam of wachtwoord is incorrect.")
    st.markdown(
    f'<div style="text-align:center"><img src="{logo}" /></div>',
    unsafe_allow_html=True
    )
    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
# --- Wanneer er geen inloggegevens ingevoerd zijn ---
if authentication_status == None:  
    st.markdown(
    f'<div style="text-align:center"><img src="{logo}" /></div>',
    unsafe_allow_html=True
    )   
    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

# --- Wanneer inloggegevens correct ingevoerd zijn
if authentication_status:

    st.markdown(
    f'<div style="text-align:center"><img src="{logo}" /></div>',
    unsafe_allow_html=True
    ) 
    
    st.markdown("<h1 style='text-align: center; color: white;'><br>Welkom!</h1>", unsafe_allow_html=True)
    

    st.sidebar.subheader(f"{name} is nu ingelogd.")
    authenticator.logout("Log uit", "sidebar")

    st.markdown("<p style='text-align: center;'><br>Deze app is gemaakt in opdracht van Westerman Multimodal Logistics B.V. en dient als eindproduct voor een afstudeeronderzoek.<br>Er kunnen geen rechten worden ontleent aan (het handelen naar) de informatie die deze app verstrekt in welke vorm dan ook.<br><br>© Bart Gerritsen 2023.</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: white;'><br><br><br>Klik op ' > ' linksbovenin om te beginnen.</h3>", unsafe_allow_html=True)
# --- EINDE VAN HET SCRIPT VOOR HOME ---