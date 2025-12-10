
import streamlit as st
import yfinance as yf
from SmartApi.smartConnect import SmartConnect
import pyotp
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import ta
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import urllib.parse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from backtesting import Backtest, Strategy
from prophet import Prophet
from streamlit_lottie import st_lottie


# --- PAGE SETUP ---
st.set_page_config(page_title="📊 Angel One Screener", layout="wide")

# --- SIDEBAR MENU ---
st.sidebar.image("https://img.icons8.com/external-flat-icons-inmotus-design/512/external-stock-stock-market-flat-icons-inmotus-design.png", width=100)
st.sidebar.title("📈 Stock Screener")
st.sidebar.caption("Powered by Angel One + Streamlit")

page = st.sidebar.radio(
    "Navigation", 
    ["🏠 Home", "🔍 Fundamentals", "📉 Charts & Indicators", "🤖 AI Prediction", "📰 News Sentiment", "📊 Advanced Analysis", "⭐ Watchlist"]
)

@st.cache_data(ttl=300)
def fetch_yahoo_indices():
    tickers = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "BANK NIFTY": "^NSEBANK",
        "MIDCAP 100": "^CNXMDCP",
        "FIN NIFTY": "^CNXFIN"
    }
    result = {}
    for name, symbol in tickers.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                last = round(data["Close"].iloc[-1], 2)
                prev_close = round(data["Close"].iloc[-2], 2)
                change = round(last - prev_close, 2)
                p_change = round((change / prev_close) * 100, 2)
                result[name] = {"last": last, "change": change, "pChange": p_change}
        except Exception as e:
            continue
    return result

@st.cache_data
def get_nse_stock_list():
    return [
        "RELIANCE", "TCS", "INFY", "SBIN", "HDFCBANK",
        "ITC", "AXISBANK", "BAJFINANCE", "HCLTECH", "LT",
        "WIPRO", "ICICIBANK", "KOTAKBANK", "ONGC", "ADANIENT",
        "POWERGRID", "NTPC", "COALINDIA", "HINDUNILVR", "ULTRACEMCO"
    ]

symbol_map = {
     "FACT": "1008",
    "RADHIKAJWE": "10343",
    "FINPIPE": "1041",
    "DYCL": "10417",
    "SATIN": "10453",
    "MUTHOOTCAP": "10415",
    "ARVSMART": "10457",
    "NIFTYBEES": "10576",
    "MARKSANS": "10579",
    "MPSLTD": "10578",
    "TCI": "10580",
    "ASAL": "10634",
    "APTECHT": "10755",
    "SYRMA": "10793",
    "RADICO": "10990",
    "INDORAMA": "10993",
    "HARSHA": "11162",
    "SHK": "11212",
    "JAIBALAJI": "11256",
    "VARDHACRLC": "11220",
    "APOLLO": "1134",
    "AFFLE": "11343",
    "TIPSFILMS": "11374",
    "RKFORGE": "11411",
    "BANKBEES": "11439",
    "TCS": "11536",
    "COFORGE": "11543",
    "DCI": "11566",
    "TRACXN": "11582",
    "HDFCSENSEX": "11593",
    "NEWGEN": "1164",
    "LALPATHLAB": "11654",
    "PATELENG": "11699",
    "SUKHJITS": "11804",
    "GOYALALUM": "11787",
    "HPIL": "11796",
    "FMGOETZE": "1190",
    "PRIMESECU": "11864",
    "DCXINDIA": "11895",
    "BIKAJI": "11966",
    "FUSION": "11932",
    "BASML": "12034",
    "AAATECH": "12533",
    "FINIETF": "12578",
    "NPBET": "12978",
    "ABB": "13",
    "SAFARI": "13035",
    "TRIVENI": "13081",
    "NAHARINDUS": "13106",
    "HDFCBANK": "1333",
    "KFINTECH": "13359",
    "BHAGERIA": "13400",
    "PFOCUS": "13496",
    "BANKETFADD": "13644",
    "GMRAIRPORT": "13528",
    "HIMATSEIDE": "1360",
    "HINDCOMPOS": "1372",
    "NAUKRI": "13751",
    "BBL": "13761",
    "SOBHA": "13826",
    "PRAENG": "13941",
    "AUTOIND": "14106",
    "REDINGTON": "14255",
    "HINDZINC": "1424",
    "SEQUENT": "14296",
    "PGIL": "14260",
    "IDEA": "14366",
    "AMDIND": "14398",
    "LEXUS": "14459",
    "BIRLANU": "1455",
    "SYNGENE": "10243",
    "AYMSYNTEX": "10285",
    "PKTEA": "10321",
    "FINCABLES": "1038",
    "LUPIN": "10440",
    "UNITDSPR": "10447",
    "GUJGASLTD": "10599",
    "TIMESGTY": "10717",
    "UNIONBANK": "10753",
    "GABRIEL": "1085",
    "TMB": "10945",
    "LIBAS": "11082",
    "SATIA": "11045",
    "LICNETFN50": "11319",
    "MAHABANK": "11377",
    "VAIBHAVGBL": "11364",
    "TTKHLTCARE": "11369",
    "PVTBANIETF": "11386",
    "CCL": "11452",
    "NDTV": "11427",
    "NV20": "11457",
    "ULTRACEMCO": "11532",
    "AMBICAAGAR": "11496",
    "LINCOLN": "11596",
    "GNFC": "1174",
    "SURYALAXMI": "11852",
    "GOKEX": "11778",
    "KANSAINER": "1196",
    "FAZE3Q": "12000",
    "HDFCPVTBAN": "12108",
    "SWSOLAR": "12489",
    "HYBRIDFIN": "12809",
    "KRISHANA": "12847",
    "ENIL": "13192",
    "KEC": "13260",
    "CENTURYPLY": "13305",
    "MALUPAPER": "13352",
    "AKASH": "13510",
    "MANUGRAPH": "13572",
    "HOVS": "13592",
    "SKYGOLD": "13631",
    "NRL": "13675",
    "GLOBALVECT": "13735",
    "GREENPANEL": "13810",
    "JINDRILL": "13875",
    "BANCOINDIA": "13880",
    "AVTNPL": "14008",
    "BTML": "14150",
    "HDFCSML250": "14233",
    "GODREJAGRO": "144",
    "EMBDL": "14450",
    "PPL": "10297",
    "JISLJALEQS": "10397",
    "MOHEALTH": "10508",
    "CONS": "10512",
    "MOM30IETF": "10585",
    "SHREEPUSHK": "10588",
    "NAVKARCORP": "10557",
    "BHARTIARTL": "10604",
    "SADBHIN": "10618",
    "NIFTYQLITY": "10690",
    "PONNIERODE": "10661",
    "SPORTKING": "10733",
    "ZENSARTECH": "1076",
    "GENESYS": "10905",
    "GRMOVER": "10871",
    "OMAXAUTO": "10922",
    "GODREJIND": "10925",
    "STCINDIA": "10948",
    "IDFCFIRSTB": "11184",
    "SUVEN": "11233",
    "UCOBANK": "11223",
    "DHANBANK": "11359",
    "UMANGDAIRY": "11387",
    "SIL": "11297",
    "SUPRAJIT": "11689",
    "JSWSTEEL": "11723",
    "ALLDIGI": "11798",
    "SSWL": "11829",
    "WELCORP": "11821",
    "VHL": "11892",
    "63MOONS": "11868",
    "GOLDIAM": "11971",
    "RML": "11975",
    "SUZLON": "12018",
    "HDFCNIFIT": "12101",
    "INOXGREEN": "12188",
    "GRASIM": "1232",
    "TEAMLEASE": "12716",
    "IIFLCAPS": "13072",
    "REPRO": "13126",
    "GALAXYSURF": "1315",
    "M&MFIN": "13285",
    "NITCO": "13300",
    "GALLANTT": "13337",
    "PIONEEREMB": "13463",
    "EMAMILTD": "13517",
    "NECCLTD": "13522",
    "VIDHIING": "13536",
    "ACLGATI": "13688",
    "INSPIRISYS": "13730",
    "GTLINFRA": "13745",
    "GESHIP": "13776",
    "NFL": "13925",
    "TANLA": "13976",
    "CONFIPET": "10238",
    "HINDUNILVR": "1394",
    "VEEDOL": "14019",
    "ZEEMEDIA": "14003",
    "GLOBAL": "1415",
    "IGPL": "14086",
    "TVSSRICHAK": "14245",
    "GOLDETF": "14286",
    "GANESHHOUC": "14339",
    "MNC": "10676",
    "LAOPALA": "14423",
    "NAHARSPING": "14440",
    "NAHARPOLY": "14445",
    "GVPTECH": "10865",
    "SIRCA": "11050",
    "NSIL": "11239",
    "JSL": "11236",
    "ZOTA": "11394",
    "POONAWALLA": "11403",
    "DATAMATICS": "11423",
    "TEXINFRA": "11549",
    "FORCEMOT": "11573",
    "INDSWFTLTD": "11602",
    "GULFPETRO": "11581",
    "PPLPHARMA": "11571",
    "XCHANGING": "11705",
    "IIFL": "11809",
    "NECLIFE": "11927",
    "YESBANK": "11915",
    "MANINDS": "11884",
    "IMPAL": "12009",
    "FCSSOFT": "11999",
    "SASKEN": "11983",
    "SCPL": "12072",
    "RUSTOMJEE": "12219",
    "PRECAM": "12603",
    "AMBUJACEM": "1270",
    "GUJALKALI": "1267",
    "SCHAEFFLER": "1011",
    "KAYA": "10276",
    "AURUM": "10310",
    "STEELCITY": "10406",
    "KSHITIJPOL": "10407",
    "SHARDAMOTR": "10530",
    "HDFCNIF100": "10633",
    "HDFCSILVER": "10876",
    "MARUTI": "10999",
    "LIQUIDBEES": "11006",
    "GARFIBRES": "1100",
    "HDFCQUAL": "11255",
    "GHCL": "1127",
    "TVSELECT": "11265",
    "MUKANDLTD": "11325",
    "GTECJAINX": "1129",
    "UPL": "11287",
    "PTC": "11355",
    "EXCELINDUS": "11471",
    "GLAXO": "1153",
    "HDFCNIFTY": "11591",
    "GUFICBIO": "11606",
    "WESTLIFE": "11580",
    "ALKEM": "11703",
    "JINDALPHOT": "11743",
    "AMBER": "1185",
    "JKPAPER": "11860",
    "ESG": "1200",
    "RENUKA": "12026",
    "MBAPL": "12686",
    "V1NSETEST": "12863",
    "DHARMAJ": "13001",
    "SILVER1": "13082",
    "HARRMALAYA": "1313",
    "EBBETF0433": "13139",
    "NITINSPIN": "13175",
    "LANDMARK": "13276",
    "MARINE": "1328",
    "HEG": "1336",
    "SDBL": "1338",
    "KELLTONTEC": "13430",
    "GPIL": "13409",
    "JKLAKSHMI": "13491",
    "BALPHARMA": "13441",
    "ACE": "13587",
    "ANDHRSUGAR": "136",
    "SHIVAMAUTO": "13756",
    "BAIDFIN": "13794",
    "DONEAR": "13839",
    "MMFL": "13844",
    "HINDPETRO": "1406",
    "NINSYS": "14194",
    "NETWORK18": "14111",
    "GANDHITUBE": "14116",
    "HDFCMID150": "14236",
    "ORIENTALTL": "14346",
    "APOLLOPIPE": "14361",
    "POCL": "14385",
    "EMAMIPAP": "10074",
    "AARON": "1030",
    "FEDERALBNK": "1023",
    "RAMASTEEL": "10300",
    "SHAREINDIA": "104",
    "OLECTRA": "10637",
    "OFSS": "10738",
    "INDIAMART": "10726",
    "MOVALUE": "10825",
    "MOVALUINAV": "10831",
    "DREAMFOLKS": "10859",
    "BANKIETF": "11037",
    "DCMNVL": "11039",
    "NATHBIOGEN": "11065",
    "AGRITECH": "11072",
    "AXISILVER": "11193",
    "INDIGO": "11195",
    "DREDGECORP": "11271",
    "PETRONET": "11351",
    "XPROINDIA": "11407",
    "APARINDS": "11491",
    "LT": "11483",
    "HDFCMOMENT": "11538",
    "CENTEXT": "11511",
    "KOTARISUG": "11647",
    "ACEINTEG": "11779",
    "SHOPERSTOP": "11813",
    "OCCL": "12231",
    "ZIMLAB": "12384",
    "DJML": "12749",
    "G1NSETEST": "12842",
    "360ONE": "13061",
    "RSYSTEMS": "13414",
    "ELIN": "13423",
    "TECHM": "13538",
    "JMFINANCIL": "13637",
    "ALICON": "13656",
    "GEECEE": "13658",
    "MANOMAY": "13753",
    "VENUSREM": "13859",
    "LIQUID1": "13950",
    "ABSLBANETF": "13987",
    "SANGHVIMOV": "14058",
    "UNOMINDA": "14154",
    "HDFCBSE500": "14230",
    "NSLNISP": "14180",
    "INDIANB": "14309",
    "HSCL": "14334",
    "GOKULAGRO": "14480",
    "THEMISMED": "14485",
    "HLVLTD": "1448",
    "ABSLLIQUID": "14518",
    "SARVESHWAR": "12913",
    "GSEC10IETF": "13143",
    "JAGRAN": "13211",
    "BLKASHYAP": "13290",
    "KEI": "13310",
    "IEL": "13303",
    "HCL-INSYS": "1327",
    "SUNTV": "13404",
    "HEROMOTOCO": "1348",
    "GRINDWELL": "13560",
    "ATLANTAA": "13585",
    "IRCTC": "13611",
    "TALBROAUTO": "13648",
    "INFOMEDIA": "13693",
    "HCC": "1375",
    "PARSVNATH": "13791",
    "MANALIPETC": "13796",
    "CREST": "13900",
    "AARTIPHARM": "13868",
    "PLASTIBLEN": "13920",
    "HBLENGINE": "13966",
    "ORIENTBELL": "14278",
    "PFC": "14299",
    "PHOENIXLTD": "14552",
    "GSLSU": "14599",
    "INSECTICID": "14657",
    "101NSETEST": "14772",
    "141NSETEST": "14781",
    "LOWVOL": "14786",
    "DECCANCE": "14838",
    "MOTILALOFS": "14947",
    "SAHYADRI": "14900",
    "PSUBNKBEES": "15032",
    "MADHAV": "15151",
    "INDIANHUME": "1530",
    "SHALPAINTS": "15342",
    "RAIN": "15337",
    "SEPC": "15308",
    "VGUARD": "15362",
    "RECLTD": "15355",
    "DPSCLTD": "15419",
    "MANKIND": "15380",
    "GODHA": "1607",
    "ARCHIDPLY": "16795",
    "LOTUSEYE": "16807",
    "JOCIL": "16927",
    "ALBERTDAVD": "17256",
    "BLS": "17279",
    "NHPC": "17400",
    "NETWEB": "17433",
    "GPTINFRA": "17685",
    "KANPRPLA": "1782",
    "LTIM": "17818",
    "AHLUCONT": "17833",
    "DLINKINDIA": "17851",
    "ABBOTINDIA": "17903",
    "AWHCL": "1797",
    "TVSSCS": "18151",
    "MANINFRA": "18226",
    "KECL": "18220",
    "HNGSNGBEES": "18284",
    "LGBBROSLTD": "18321",
    "BIGBLOC": "18431",
    "RBLBANK": "18391",
    "LAURUSLABS": "19234",
    "KOPRAN": "1919",
    "MONIFTY500": "19237",
    "AHLEAST": "19438",
    "BOROLTD": "19401",
    "UGARSUGAR": "19578",
    "BSE": "19585",
    "AMNPLST": "19867",
    "EGOLD": "19933",
    "ESAFSFB": "19878",
    "INTENTECH": "20071",
    "URJA": "20203",
    "OBEROIRLTY": "20242",
    "IOLCP": "20413",
    "HOMEFIRST": "2056",
    "INDIASHLTR": "20556",
    "JAMNAAUTO": "20778",
    "PSPPROJECT": "20877",
    "BFINVEST": "21113",
    "KICL": "21119",
    "MARALOVER": "2112",
    "ICICIGI": "21770",
    "RHFL": "21733",
    "ORICONENT": "10159",
    "ADANIENSOL": "10217",
    "CSLFINANCE": "10350",
    "DGCONTENT": "10346",
    "UTISENSETF": "10515",
    "MODTHREAD": "2241",
    "POWERMECH": "10473",
    "PNB": "10666",
    "SILVERADD": "10761",
    "INFRAIETF": "10723",
    "JSFB": "22663",
    "NUCLEUS": "10791",
    "SURANAT&P": "10700",
    "MOQLTYINAV": "10830",
    "CAPITALSFB": "22675",
    "MRF": "2277",
    "INFOBEAN": "11027",
    "KSOLVES": "11060",
    "CREATIVE": "11155",
    "HDFCGROWTH": "11241",
    "UNIENTER": "11293",
    "IGL": "11262",
    "LUXIND": "11301",
    "HAL": "2303",
    "GICHSGFIN": "1139",
    "SPANDANA": "11435",
    "KRITINUT": "11432",
    "HDFCLOWVOL": "11547",
    "NCC": "2319",
    "ASTRAMICRO": "11618",
    "JPPOWER": "11763",
    "DWARKESH": "11667",
    "GMBREW": "1168",
    "GODFRYPHLP": "1181",
    "GLAND": "1186",
    "MEDANTA": "11956",
    "SATINDLTD": "12015",
    "AURIONPRO": "12022",
    "DOLATALGO": "12124",
    "KAYNES": "12092",
    "JNKINDIA": "23621",
    "GREAVESCOT": "1235",
    "AADHARHFC": "23729",
    "GSFC": "1247",
    "ABSLPSE": "23806",
    "TBOTEK": "23740",
    "GODIGIT": "23799",
    "NELCO": "2388",
    "ROUTE": "128",
    "ORIENTCER": "13111",
    "UNIPARTS": "13057",
    "QUICKHEAL": "13116",
    "PVRINOX": "13147",
    "ROHLTD": "13182",
    "TOP100CASE": "24081",
    "NEULANDLAB": "2406",
    "COMMOIETF": "13198",
    "INDOTECH": "13275",
    "GSPL": "13197",
    "AFSL": "13288",
    "KKCL": "13381",
    "RELAXO": "24225",
    "VOLTAMP": "13577",
    "RADIANTCMS": "13580",
    "ANANTRAJ": "13620",
    "ESTER": "24265",
    "SAH": "13689",
    "ELECON": "13643",
    "LIQUIDSHRI": "24374",
    "VISHWARAJ": "13702",
    "ASHIANA": "24403",
    "NKIND": "2439",
    "KARMAENG": "24385",
    "FLUOROCHEM": "13750",
    "KUANTUM": "13870",
    "LTFOODS": "13816",
    "HIRECT": "13890",
    "MAITHANALL": "24538",
    "HINDOILEXP": "1403",
    "SARLAPOLY": "14043",
    "ASMS": "14064",
    "PTL": "14101",
    "AVANTEL": "24610",
    "SAGCEM": "14068",
    "PITTIENG": "14134",
    "RUBFILA": "24730",
    "RPEL": "24735",
    "SMSPHARMA": "14329",
    "NCLIND": "14490",
    "MANCREDIT": "24823",
    "INTERARCH": "24909",
    "TRITURBINE": "25584",
    "JISLDVREQS": "25684",
    "RUPA": "25724",
    "POLYMED": "25718",
    "CHEMCON": "270",
    "GICRE": "277",
    "CHEMFAB": "2799",
    "JUBLINGREA": "2783",
    "DUCON": "28952",
    "LAXMIDENTL": "29171",
    "AGARWALEYE": "29452",
    "MOHITIND": "29423",
    "SAIL": "2963",
    "GROWWN200": "29872",
    "VIVIDHA": "29877",
    "ATULAUTO": "30023",
    "WEBELSOLAR": "14602",
    "SRHHYPOLTD": "14582",
    "ACCURACY": "1465",
    "GOACARBON": "14687",
    "NAGREEKEXP": "14702",
    "AXSENSEX": "14742",
    "151NSETEST": "14782",
    "041NSETEST": "14755",
    "OMAXE": "14853",
    "CENTRALBK": "14894",
    "LAL": "14924",
    "INDOWIND": "14952",
    "CSBBANK": "14966",
    "CENTUM": "14982",
    "PSUBANK": "15061",
    "ADANIPORTS": "15083",
    "EDELWEISS": "15119",
    "INDHOTEL": "1512",
    "HITECHCORP": "15161",
    "KAUSHALYA": "15136",
    "KIRLPNU": "15180",
    "INDIAGLYCO": "1521",
    "RKEC": "1547",
    "RVHL": "1565",
    "MKPL": "16736",
    "GOKUL": "16705",
    "VINYLINDIA": "16821",
    "SINDHUTRAD": "16859",
    "ASHOKAMET": "16943",
    "SUMICHEM": "17105",
    "NEXT50": "17181",
    "CYIENTDLM": "17187",
    "PSUBANKADD": "17616",
    "MOLDTECH": "17625",
    "VIJIFIN": "17691",
    "DBCORP": "17881",
    "HINDCOPPER": "17939",
    "BEARDSELL": "17933",
    "ADVENZYMES": "18039",
    "MINDTECK": "18049",
    "JUBLFOOD": "18096",
    "DBREALTY": "18124",
    "HATHWAY": "18154",
    "PYRAMID": "18250",
    "SPAL": "18252",
    "IVZINGOLD": "18292",
    "TCPLPACK": "184",
    "ISFT": "18479",
    "NIRAJISPAT": "18840",
    "SETFNIF50": "10176",
    "KAPSTON": "18967",
    "BALKRISHNA": "10181",
    "UNIVASTU": "18977",
    "JSWINFRA": "19020",
    "SILLYMONKS": "19097",
    "AXISBNKETF": "1044",
    "UTINIFTETF": "10511",
    "RAMRAT": "10485",
    "MOMENTUM": "10693",
    "HDFCNEXT50": "10619",
    "OAL": "10768",
    "MOQUALITY": "10822",
    "CANBK": "10794",
    "ITETF": "19633",
    "LIQUIDSBI": "19705",
    "PRICOLLTD": "19631",
    "JUNIORBEES": "10939",
    "DIVISLAB": "10940",
    "WELSPUNLIV": "11253",
    "BALAXI": "11309",
    "TVTODAY": "11275",
    "BIOCON": "11373",
    "SALZERELEC": "11399",
    "GIPCL": "1145",
    "EMIL": "11530",
    "SALSTEEL": "11634",
    "WELENT": "11626",
    "NTPC": "11630",
    "NARMADA": "11627",
    "INDOCO": "11677",
    "JBMA": "11655",
    "SAKSOFT": "11794",
    "EVEREADY": "11782",
    "NH": "11840",
    "ICIL": "11987",
    "HTMEDIA": "11979",
    "GENUSPOWER": "11905",
    "FIVESTAR": "12032",
    "HNDFDS": "12173",
    "CASTROLIND": "1250",
    "GFLLIMITED": "1289",
    "AIAENG": "13086",
    "EKC": "13091",
    "KERNEX": "13121",
    "SULA": "13218",
    "SOLARINDS": "13332",
    "JKCEMENT": "13270",
    "SGL": "13367",
    "RATNAMANI": "13451",
    "ALLCARGO": "13501",
    "SELAN": "13598",
    "HINDALCO": "1363",
    "MADHUCON": "13671",
    "SHYAMTEL": "13740",
    "CANTABIL": "20160",
    "PARACABLES": "13951",
    "ASHAPURMIN": "203",
    "JASH": "13982",
    "LUMAXTECH": "14014",
    "AXISGOLD": "20532",
    "BBNPPGOLD": "20547",
    "BAJEL": "20531",
    "WANBURY": "14063",
    "AGI": "1412",
    "INOXINDIA": "20607",
    "PENIND": "20621",
    "SIYSIL": "14096",
    "TIMKEN": "14198",
    "HUBTOWN": "14203",
    "SCHAND": "20698",
    "TTL": "14314",
    "FSL": "14304",
    "SURAJEST": "20822",
    "KIRLOSENG": "20936",
    "DIVGIITTS": "14479",
    "SANOFI": "1442",
    "BALAMINES": "14501",
    "TEJASNET": "21131",
    "PALASHSECU": "21383",
    "TATSILV": "21428",
    "THEINVEST": "21427",
    "COCHINSHIP": "21508",
    "MODIRUBBER": "2229",
    "WORTH": "22277",
    "LOVABLE": "22415",
    "MAXIND": "22428",
    "BANDHANBNK": "2263",
    "PARKHOTELS": "22649",
    "MRPL": "2283",
    "HLEGLAS": "2289",
    "ARE&M": "100",
    "MUKKA": "22980",
    "PLATIND": "22930",
    "GODREJCP": "10099",
    "MANORAMA": "10227",
    "UFLEX": "1053",
    "KRBL": "10577",
    "FOSECOIND": "1073",
    "SUMIT": "11140",
    "KBCGLOBAL": "2354",
    "HDFCVALUE": "11260",
    "LICNETFSEN": "11441",
    "PILANIINVS": "11445",
    "NIITLTD": "11522",
    "NAVNETEDUL": "2385",
    "GTL": "1162",
    "CONSOFINVT": "11731",
    "SANDHAR": "2397",
    "MANGALAM": "11817",
    "GRANULES": "11872",
    "GEOJITFSL": "11896",
    "KRONOX": "24025",
    "MSPL": "11919",
    "MCLEODRUSS": "11943",
    "SYNCOMF": "11992",
    "PRAXIS": "1204",
    "ACI": "12024",
    "PEL": "2412",
    "NILKAMAL": "2421",
    "AFIL": "24159",
    "SYMPHONY": "24190",
    "BBTCL": "12153",
    "OMINFRAL": "24231",
    "OSIAHYPER": "12635",
    "VSTTILLERS": "24292",
    "OILIETF": "24533",
    "N1NSETEST": "12848",
    "11NSETEST": "12841",
    "VIMTALABS": "13101",
    "OLAELEC": "24777",
    "CELEBRITY": "13162",
    "SURAJLTD": "24966",
    "UTTAMSUGAR": "13376",
    "KAMDHENU": "13457",
    "GALAPREC": "25134",
    "OSWALAGRO": "2514",
    "MUNJALAU": "13511",
    "TOP10ADD": "25171",
    "KROSS": "25252",
    "BALAJEE": "25203",
    "FIEMIND": "13710",
    "BFUTILITIE": "14567",
    "DCBBANK": "13725",
    "RUCHIRA": "13821",
    "TORNTPOWER": "13786",
    "NAVINFLUOR": "14672",
    "HGS": "14712",
    "021NSETEST": "14751",
    "131NSETEST": "14778",
    "TIIL": "14223",
    "USK": "14871",
    "ADSL": "14813",
    "PAGEIND": "14413",
    "ASTRAL": "14418",
    "SHIVAMILLS": "1497",
    "ASIANTILES": "14889",
    "KDDL": "14908",
    "GOLDBEES": "14428",
    "JKTYRE": "14435",
    "CCCL": "14992",
    "MOGSEC": "1507",
    "BANSWRAS": "14511",
    "ICRA": "14523",
    "PSUBNKIETF": "14584",
    "TGBHOTELS": "14607",
    "FORTIS": "14592",
    "KNRCON": "15283",
    "031NSETEST": "14753",
    "DLF": "14732",
    "061NSETEST": "14762",
    "IDBI": "1476",
    "NELCAST": "14761",
    "TARC": "1581",
    "MAGNUM": "14957",
    "PRINCEPIPE": "16045",
    "IOC": "1624",
    "KSCL": "14972",
    "MANGCHEFER": "15007",
    "EBBETF0430": "16253",
    "LINDEINDIA": "1627",
    "ENERGYDEV": "15049",
    "CERA": "15039",
    "KOLTEPATIL": "15124",
    "ITC": "1660",
    "INDIACEM": "1515",
    "TARIL": "15174",
    "COLPAL": "15141",
    "AMBIKCO": "15234",
    "UJJIVANSFB": "15228",
    "HGINFRA": "1672",
    "GSS": "15347",
    "HINDWAREAP": "15883",
    "LPDC": "16863",
    "APCOTEXIND": "154",
    "INGERRAND": "1597",
    "ZFCVINDIA": "16915",
    "PAVNAIND": "16192",
    "VAISHALI": "16589",
    "BAJAJ-AUTO": "16669",
    "JAYBARMARU": "1708",
    "UBL": "16713",
    "KOKUYOCMLN": "16827",
    "IKIO": "16822",
    "IDEAFORGE": "17140",
    "CROMPTON": "17094",
    "SENCO": "17271",
    "AARTECH": "17145",
    "MHRIL": "17333",
    "EXPLEOSOL": "17486",
    "PVTBANKADD": "17576",
    "ZYDUSWELL": "17635",
    "JINDALPOLY": "1756",
    "UNIVPHOTO": "17659",
    "MIDSELIETF": "17702",
    "ASTEC": "17728",
    "POKARNA": "17651",
    "DEN": "17722",
    "SHILPAMED": "17752",
    "TREL": "17795",
    "COHANCE": "17945",
    "QUAL30IETF": "17808",
    "GEEKAYWIRE": "17922",
    "VASCONEQ": "18110",
    "THANGAMAYL": "18118",
    "EMMBI": "18142",
    "WHIRLPOOL": "18011",
    "NIFTY1": "18102",
    "SRGHFL": "18119",
    "KARURVYSYA": "1838",
    "POWERINDIA": "18457",
    "NIFMID150": "18347",
    "TEXMOPIPES": "18214",
    "KCP": "1841",
    "MITTAL": "18562",
    "MARATHON": "18659",
    "KIRLOSBROS": "18581",
    "SJVN": "18883",
    "SUNDARAM": "18931",
    "KINGFA": "18944",
    "MANAPPURAM": "19061",
    "YATRA": "18760",
    "MVGJL": "19015",
    "CUMMINSIND": "1901",
    "KIOCL": "19126",
    "AARVI": "19073",
    "EMAMIREAL": "19277",
    "NBIFIN": "19111",
    "SFL": "19184",
    "IRMENERGY": "19597",
    "TCIEXP": "19223",
    "MOM50": "19289",
    "CAPTRUST": "19447",
    "IITL": "19525",
    "ADFFOODS": "19761",
    "RADIOCITY": "19877",
    "SECMARK": "19496",
    "DMART": "19913",
    "AHLADA": "2004",
    "MAXESTATES": "19646",
    "AKSHARCHEM": "20178",
    "BLUEJET": "19686",
    "KILITCH": "19937",
    "LYKALABS": "2028",
    "CLEDUCATE": "20223",
    "SHAH": "20296",
    "TATATECH": "20293",
    "M&M": "2031",
    "FLAIR": "20372",
    "ASKAUTOLTD": "20152",
    "WIPL": "20481",
    "GOLDSHARE": "14535",
    "ESILVER": "20257",
    "ZENITHSTL": "14562",
    "TARMAT": "14771",
    "SPARC": "14788",
    "ALPA": "14848",
    "KPRMILL": "14912",
    "NAGREEKCAP": "14942",
    "ASTERDM": "1508",
    "WCIL": "25403",
    "JYOTHYLAB": "15146",
    "CORDSCABLE": "15271",
    "GOLDETFADD": "15356",
    "IRB": "15313",
    "MANBA": "25597",
    "GROWWDEFNC": "25758",
    "NESCO": "15409",
    "TITAGARH": "15414",
    "AVG": "15589",
    "MGEL": "1593",
    "FAIRCHEMOR": "1614",
    "IONEXCHANG": "1630",
    "PTCIL": "16682",
    "RPGLIFE": "16725",
    "ITI": "1675",
    "AVL": "25984",
    "INDBANK": "16933",
    "SHARIABEES": "17044",
    "PATANJALI": "17029",
    "GATECHDVR": "17",
    "ATUL": "263",
    "SETFGOLD": "17272",
    "PIDILITIND": "2664",
    "HEXATRADEX": "27008",
    "ASPINWALL": "17270",
    "ARMANFIN": "17263",
    "ACMESOLAR": "27061",
    "LLOYDSME": "17313",
    "NTPCGREEN": "27176",
    "DAMODARIND": "17541",
    "CONTROLPR": "17477",
    "MGL": "17534",
    "LIQUID": "17572",
    "RTNINDIA": "27297",
    "YATHARTH": "17738",
    "SARDAEN": "17758",
    "NESTLEIND": "17963",
    "JYOTISTRUC": "1802",
    "KAJARIACER": "1808",
    "AUTOAXLES": "278",
    "EASEMYTRIP": "2792",
    "YASHO": "18131",
    "KPIL": "1814",
    "RAJSREESUG": "2809",
    "ORIENTLTD": "18208",
    "HDFCLIQUID": "18279",
    "KANORICHEM": "1835",
    "ZAGGLE": "18608",
    "HPL": "18679",
    "MAZDA": "18866",
    "RAJRATAN": "18962",
    "KOTAKBANK": "1922",
    "QUADFUTURE": "29087",
    "GOLDIETF": "19679",
    "GPPL": "19731",
    "WELINV": "19800",
    "SUPREME": "29439",
    "LICHSGFIN": "1997",
    "NEXT30ADD": "29464",
    "AJAXENGG": "29624",
    "BAJAJINDEF": "29678",
    "SURYODAY": "2970",
    "DOMS": "20551",
    "CHENNPETRO": "2049",
    "SUPERHOUSE": "20616",
    "GUJAPOLLO": "14677",
    "AKZOINDIA": "1467",
    "KMSUGAR": "14667",
    "161NSETEST": "14784",
    "PSB": "21001",
    "IFBIND": "1485",
    "IFBAGRO": "1482",
    "ERIS": "21154",
    "SILVERETF": "15085",
    "BHAGYANGR": "20776",
    "SVLL": "15121",
    "RHL": "15157",
    "MAHSCOOTER": "2085",
    "HAPPYFORGE": "20854",
    "CEATLTD": "15254",
    "RPOWER": "15259",
    "VIRINCHI": "15295",
    "ONMOBILE": "15278",
    "BANG": "15303",
    "MANGLMCEM": "2106",
    "SURANASOL": "21077",
    "HCG": "15555",
    "ACL": "15420",
    "CDSL": "21174",
    "AVADHSUGAR": "21406",
    "GMMPFAUDLR": "1570",
    "ATALREAL": "15649",
    "UTINEXT50": "21478",
    "MFSL": "2142",
    "GILLETTE": "1576",
    "BECTORFOOD": "1628",
    "TEXRAIL": "21828",
    "DPWIRES": "16900",
    "IEX": "220",
    "DHUNINV": "22233",
    "SHRIPISTON": "17186",
    "VINATIORGA": "17364",
    "AJMERA": "17307",
    "SETF10GILT": "17395",
    "GLOBUSSPR": "17424",
    "ADANIPOWER": "17388",
    "NIF10GETF": "22524",
    "SENSEXADD": "17613",
    "RTNPOWER": "17520",
    "LTGILTBEES": "17700",
    "NIITMTS": "17747",
    "ALOKINDS": "17675",
    "UNITEDTEA": "17999",
    "KAKATCEM": "1811",
    "GOPAL": "23066",
    "KAMATHOTEL": "1826",
    "MOREALTY": "23184",
    "PERSISTENT": "18365",
    "AIIL": "23553",
    "RRKABEL": "18566",
    "BSLGOLDETF": "23804",
    "NUVAMA": "18721",
    "KALAMANDIR": "18755",
    "STYRENIX": "19",
    "VBL": "18921",
    "AARTISURF": "19233",
    "KREBSBIO": "1937",
    "BHANDARI": "19556",
    "PRAKASHSTL": "19599",
    "ALPHAETF": "19640",
    "MASFIN": "199",
    "LMW": "1979",
    "LIBERTSHOE": "1994",
    "GREENPOWER": "20110",
    "SASTASUNDR": "20063",
    "LUMAXIND": "2018",
    "GANDHAR": "20303",
    "INDIGOPNTS": "2048",
    "VTL": "2073",
    "JINDWORLD": "20642",
    "ZEELEARN": "20852",
    "AZAD": "20905",
    "STARCEMENT": "21091",
    "AXISNIFTY": "21252",
    "TATAGOLD": "21401",
    "ABCAPITAL": "21614",
    "COMSYN": "21578",
    "ONGC": "2475",
    "APEX": "21623",
    "DCAL": "21704",
    "YAARI": "24999",
    "A2ZINFRA": "20906",
    "STYLEBAAZA": "25114",
    "ARKADE": "25398",
    "HUHTAMAKI": "2562",
    "VIKASECO": "25756",
    "GTPL": "21195",
    "AUBANK": "21238",
    "AFFORDABLE": "25855",
    "DYNPRO": "21314",
    "WAAREEENER": "25907",
    "3PLAND": "2595",
    "MOM100": "21423",
    "GANGESSECU": "21399",
    "SMSLIFE": "21551",
    "SWANENERGY": "27095",
    "ARVEE": "2814",
    "IKS": "28125",
    "ANURAS": "2829",
    "LXCHEM": "2841",
    "CRAFTSMAN": "2854",
    "SANATHAN": "28805",
    "MOSCHIP": "29459",
    "PROTEAN": "29472",
    "NATCAPSUQ": "29583",
    "LOWVOLIETF": "21254",
    "DIGIDRIVE": "21247",
    "MAGADSUGAR": "21392",
    "JYOTICNC": "21334",
    "OPTIEMUS": "21469",
    "CHOLAHLDNG": "21740",
    "CAPACITE": "21749",
    "DIAMONDYD": "21840",
    "SBILIFE": "21808",
    "INDTERRAIN": "21957",
    "ACC": "22",
    "EBBETF0431": "22239",
    "MAXHEALTH": "22377",
    "HDFCNIFBAN": "22433",
    "MON100": "22739",
    "LASA": "21713",
    "JGCHEM": "23056",
    "HFCL": "21951",
    "NOVAAGRI": "22477",
    "HEIDELBERG": "2316",
    "SMCGLOBAL": "2320",
    "181NSETEST": "23447",
    "GPTHEALTH": "22846",
    "JUNIPER": "22824",
    "BALUFORGE": "23607",
    "MUNJALSHOW": "2307",
    "SCILAL": "23127",
    "IXIGO": "24067",
    "STANLEY": "24226",
    "ABGSEC": "24404",
    "SANSTAR": "24582",
    "INVENTURE": "24870",
    "SSDL": "24856",
    "ORIENTHOT": "2493",
    "ADANIENT": "25",
    "PGEL": "25358",
    "PGHH": "2535",
    "BHARTIHEXA": "23489",
    "UYFINCORP": "25626",
    "SRD": "25737",
    "MINDACORP": "25897",
    "DIVOPPBEES": "2636",
    "AMJLAND": "2598",
    "KPEL": "27079",
    "MIDSMALL": "23855",
    "RACLGEAR": "27171",
    "MTARTECH": "2709",
    "ANUHPHR": "28314",
    "CARERATING": "29113",
    "RIIL": "2912",
    "ABDL": "24308",
    "PCJEWELLER": "29124",
    "VRAJ": "24321",
    "SBISILVER": "24366",
    "EVINDIA": "24461",
    "KALYANKJIL": "2955",
    "MIDHANI": "2463",
    "HILTON": "14627",
    "FIRSTCRY": "24814",
    "METALIETF": "24861",
    "ORIENTPPR": "2496",
    "TIMETECHNO": "14707",
    "051NSETEST": "14758",
    "121NSETEST": "14777",
    "071NSETEST": "14764",
    "BCLIND": "2513",
    "GSEC10YEAR": "14938",
    "CIEINDIA": "14937",
    "TDPOWERSYS": "25178",
    "MAANALU": "15017",
    "APLLTD": "25328",
    "BVCL": "15073",
    "PATINTLOG": "15219",
    "ECLERX": "15179",
    "PPAP": "15209",
    "ARIES": "15204",
    "LICNFNHGP": "15229",
    "PAISALO": "25468",
    "NAHARCAP": "15360",
    "RACE": "15391",
    "ATAM": "15404",
    "LICMFGOLD": "25640",
    "BHARATWIRE": "16123",
    "APOLLOHOSP": "157",
    "INFIBEAM": "16249",
    "TATAINVEST": "1621",
    "APOLLOTYRE": "163",
    "MMP": "16395",
    "VAL30IETF": "25851",
    "HYUNDAI": "25844",
    "SIGMA": "16658",
    "GHCLTEXTIL": "16696",
    "BAJAJFINSV": "16675",
    "DIGISPICE": "16683",
    "DRCSYSTEMS": "2645",
    "IVP": "1681",
    "ARTEMISMED": "16913",
    "VSSL": "27067",
    "SWIGGY": "27066",
    "KMEW": "27084",
    "PRECOT": "2711",
    "THYROCARE": "17032",
    "HMAAGRO": "17053",
    "MIDCAPIETF": "17152",
    "PRECWIRE": "2717",
    "BUTTERFLY": "2716",
    "JBCHEPHARM": "1726",
    "SAILIFE": "27839",
    "SUNTECK": "17641",
    "MAHEPC": "17603",
    "QUESS": "17704",
    "GILLANDERS": "17839",
    "ATL": "17778",
    "IGIL": "28378",
    "GODREJPROP": "17875",
    "JSWENERGY": "17869",
    "BAYERCROP": "17927",
    "VENTIVE": "28847",
    "TRF": "17987",
    "SBICARD": "17971",
    "CARRARO": "28879",
    "RATNAVEER": "18423",
    "NIBE": "29487",
    "CUPID": "18520",
    "GNA": "18571",
    "LTTS": "18564",
    "NIF100BEES": "29577",
    "REPCOHOME": "29598",
    "JUBLCPL": "29590",
    "CARYSIL": "1879",
    "SAKHTISUG": "2969",
    "KSL": "18889",
    "NIBL": "29733",
    "PODDARMENT": "19036",
    "JUSTDIAL": "29962",
    "ITBEES": "19084",
    "UDS": "19113",
    "ROLLT": "19104",
    "DAVANGERE": "21794",
    "SUNDARMHLD": "2183",
    "NIFTYBETF": "21959",
    "ORISSAMINE": "19931",
    "DEEPAKNTR": "19943",
    "INFRABEES": "20072",
    "UNIDT": "201",
    "ARIHANTSUP": "20159",
    "RAMKY": "20134",
    "WABAG": "20188",
    "PRESTIGE": "20302",
    "NRAIL": "20314",
    "COALINDIA": "20374",
    "DOLLAR": "20560",
    "MCL": "22360",
    "ITIETF": "22365",
    "MOTISONS": "20788",
    "AROGRANITE": "14557",
    "ADVANIHOTR": "14745",
    "081NSETEST": "14767",
    "REPL": "1480",
    "GOLD1": "14858",
    "MOREPENLAB": "2259",
    "PURVA": "14922",
    "RBA": "1494",
    "POWERGRID": "14977",
    "RELIGARE": "15068",
    "AVALON": "15058",
    "BLAL": "15067",
    "BAJAJELEC": "15034",
    "DVL": "15214",
    "MANAKSIA": "15199",
    "SMALLCAP": "22832",
    "JKIL": "15266",
    "HERCULES": "15288",
    "INFY": "1594",
    "MURUDCERA": "2313",
    "MOSMALL250": "23181",
    "SILVRETF": "16777",
    "QNIFTY": "16819",
    "20MICRONS": "16921",
    "MAWANASUG": "17022",
    "ITETFADD": "17207",
    "JAYSREETEA": "1720",
    "PAKKA": "17197",
    "SALONA": "17351",
    "ASIANPAINT": "236",
    "NV20IETF": "17475",
    "AMRUTANJAN": "17547",
    "GROBTEA": "17770",
    "REFEX": "17764",
    "LLOYDSENGG": "17801",
    "BOROSCI": "24014",
    "REDTAPE": "17859",
    "SBFC": "18026",
    "IVZINNIFTY": "24217",
    "DBL": "18086",
    "ARENTERP": "181",
    "RELIABLE": "24378",
    "NDGL": "18371",
    "DIACABS": "18543",
    "DIAMINESQ": "18644",
    "ICICIPRULI": "18652",
    "SIGNATURE": "18743",
    "UNIECOM": "24809",
    "IDFNIFTYET": "18783",
    "SENSEXETF": "19224",
    "IMFA": "19235",
    "ITDC": "19299",
    "ARVIND": "193",
    "MULTICAP": "25080",
    "PREMIERENE": "25049",
    "PLAZACABLE": "19458",
    "BAJAJCON": "19531",
    "EUREKAFORB": "25162",
    "TRENT": "1964",
    "SANOFICONR": "25222",
    "CELLO": "19795",
    "LGHL": "19876",
    "NORTHARC": "25426",
    "ASHOKA": "20182",
    "FEDFINA": "20322",
    "CGCL": "20329",
    "DSSL": "25690",
    "APCL": "20453",
    "PRUDMOULI": "20527",
    "GRAVITA": "20534",
    "SEAMECLTD": "2616",
    "PEARLPOLY": "2610",
    "PFIZER": "2643",
    "POLYPLEX": "2687",
    "PRAJIND": "2705",
    "NIVABUPA": "27097",
    "SURAKSHA": "27334",
    "RCF": "2866",
    "MAYURUNIQ": "28906",
    "UNIMECH": "28960",
    "ONESOURCE": "29224",
    "MOIL": "20830",
    "HUDCO": "20825",
    "MUFTI": "20878",
    "MCLOUD": "29482",
    "NIFTYIETF": "29553",
    "SANDESH": "2988",
    "SIS": "21501",
    "DIXON": "21690",
    "MOCAPITAL": "30065",
    "CASHIETF": "30139",
    "SAURASHCEM": "3018",
    "BHARATSE": "30244",
    "TOTAL": "22423",
    "TRANSWORLD": "3129",
    "NDL": "31258",
    "BLSE": "22566",
    "HONDAPOWER": "3138",
    "VALIANTORG": "330",
    "LODHA": "3220",
    "SIGNPOST": "22656",
    "MOTOGENFIN": "2268",
    "MODISONLTD": "3316",
    "ADOR": "34",
    "BALMLAWRIE": "338",
    "HONAUT": "3417",
    "TATAMOTORS": "3456",
    "TATACONSUM": "3432",
    "TNTELE": "3512",
    "GOLDCASE": "22901",
    "UNICHEMLAB": "3579",
    "VESUVIUS": "3676",
    "SOLARA": "3672",
    "EXICOM": "22947",
    "VLSFINANCE": "3715",
    "WALCHANNAG": "3736",
    "TATACOMM": "3721",
    "NATCOPHARM": "3918",
    "IWEL": "3776",
    "WILLAMAGOR": "3772",
    "PVSL": "23142",
    "NAVA": "4014",
    "PANACEABIO": "4055",
    "GOCLCORP": "3963",
    "SANGAMIND": "4184",
    "HDFCAMC": "4244",
    "JAYNECOIND": "2331",
    "LIQUIDADD": "23407",
    "PUNJABCHEM": "4344",
    "INDIANCARD": "4422",
    "GLOSTERLTD": "23590",
    "ATLASCYCLE": "4568",
    "JCHAC": "4491",
    "SONACOMS": "4684",
    "SHYAMMETL": "4693",
    "KCPSUGIND": "4809",
    "SAREGAMA": "4892",
    "BFSI": "5220",
    "BOMDYEING": "513",
    "EXXARO": "5352",
    "AAREYDRUGS": "5245",
    "INTLCONV": "5606",
    "BSLSENETFG": "5957",
    "CUB": "5701",
    "CENTENKA": "619",
    "NIFTYETF": "6353",
    "HEALTHY": "6297",
    "APOLSINHOT": "6302",
    "SONATSOFTW": "6596",
    "POLICYBZR": "6656",
    "IVC": "6711",
    "GANECOS": "6944",
    "VISHNU": "6908",
    "DICIND": "703",
    "SHIVALIK": "7016",
    "JMA": "7109",
    "STARHEALTH": "7083",
    "SBIETFPB": "722",
    "HECPROJECT": "7425",
    "HAVISHA": "7457",
    "TINNARUBR": "756020",
    "PRIMO": "756925",
    "MONARCH": "7679",
    "RAMCOSYS": "7851",
    "SILVERIETF": "7942",
    "LFIC": "7902",
    "AWL": "8110",
    "BLBLIMITED": "8132",
    "BSHSL": "809",
    "MEGASTAR": "8153",
    "SOFTTECH": "8266",
    "GSEC5IETF": "8342",
    "TVSMOTOR": "8479",
    "MITCON": "8469",
    "GATEWAY": "8510",
    "PNBGILTS": "8736",
    "ROTO": "9049",
    "EICHERMOT": "910",
    "EQUITASBNK": "913",
    "ORCHPHARMA": "926",
    "SILVERTUC": "9422",
    "SKIPPER": "9428",
    "ETHOSLTD": "9750",
    "AETHER": "9810",
    "EPL": "981",
    "AIRAN": "9897",
    "ALANKIT": "9921",
    "RAILTEL": "2431",
    "CONSUMBEES": "2435",
    "DHANUKA": "24409",
    "AGIIL": "24445",
    "OILCOUNTUB": "2472",
    "AKUMS": "24715",
    "CEIGALL": "24742",
    "ONWARDTEC": "2481",
    "ORIENTTECH": "24961",
    "BODALCHEM": "25017",
    "RAYMONDLSL": "25073",
    "PANAMAPET": "25392",
    "VASWANI": "25340",
    "ESSARSHPNG": "25634",
    "METAL": "25744",
    "GARUDA": "25800",
    "EMULTIMQ": "25996",
    "OCCLLTD": "25918",
    "MASTERTR": "27047",
    "PRAKASH": "2708",
    "TRANSRAILL": "28714",
    "DAMCAPITAL": "28662",
    "SENORES": "28888",
    "SGLTL": "29075",
    "RICOAUTO": "2909",
    "ITCHOTELS": "29251",
    "VMART": "29284",
    "QPOWER": "29711",
    "NAZARA": "2987",
    "GUJTHEM": "29731",
    "SANGHIIND": "2997",
    "BSE500IETF": "3001",
    "RKDL": "20950",
    "SUNCLAY": "20956",
    "INNOVACAP": "21062",
    "SUMMITSEC": "21275",
    "ASHOKLEY": "212",
    "SALASAR": "21362",
    "BDL": "2144",
    "HINDMOTORS": "21676",
    "MATRIMONY": "21726",
    "BOSCHLTD": "2181",
    "AKG": "2176",
    "BANKBETF": "21986",
    "NDRAUTO": "22255",
    "ALPL30IETF": "22344",
    "NIF5GETF": "22521",
    "HDFCPSUBK": "22595",
    "HEALTHADD": "22592",
    "RPTECH": "22670",
    "LICNMID100": "22722",
    "ENTERO": "22717",
    "IRISDOREME": "2275",
    "LANCORHOL": "23027",
    "CPSEETF": "2328",
    "SRM": "23402",
    "171NSETEST": "23446",
    "INDGN": "23693",
    "AWFIS": "23864",
    "MID150CASE": "24077",
    "VADILALIND": "24196",
    "PIIND": "24184",
    "BANSALWIRE": "24386",
    "EMCURE": "24398",
    "SBINEQWETF": "24524",
    "CENTRUM": "2454",
    "FILATEX": "24532",
    "BSLNIFTY": "24781",
    "MODEFENCE": "24944",
    "WINDMACHIN": "24969",
    "ECOSMOBLTY": "25060",
    "XTGLOBAL": "25235",
    "PNGJL": "25312",
    "NIRAJ": "255",
    "BANKPSU": "25725",
    "KRN": "25643",
    "LLOYDSENT": "25807",
    "VINCOFE": "25835",
    "GROWWGOLD": "25872",
    "AFCONS": "25977",
    "IT": "2627",
    "PCBL": "2649",
    "LEMONTREE": "2606",
    "TBZ": "27037",
    "ABINFRA": "27020",
    "SAGILITY": "27052",
    "LIQUIDPLUS": "27075",
    "BLACKBUCK": "27144",
    "SAMPANN": "27303",
    "RALLIS": "2816",
    "VMM": "27969",
    "MOBIKWIK": "28046",
    "RANEHOLDIN": "2844",
    "AONELIQUID": "30472",
    "CEWATER": "28764",
    "HITECH": "2868",
    "BAJAJHIND": "308",
    "LYPSAGEMS": "31468",
    "KITEX": "28899",
    "ECAPINSURE": "29019",
    "SUNDARMFIN": "3339",
    "DDEVPLSTIK": "29098",
    "SUPREMEIND": "3363",
    "TAINWALCHM": "3396",
    "INDUSTOWER": "29135",
    "RUBYMILLS": "2939",
    "ROML": "359",
    "GROWWRAIL": "29527",
    "KHAITANLTD": "3912",
    "WSTCSTPAPR": "3799",
    "ZEEL": "3812",
    "UNIONGOLD": "29687",
    "CORALFINAC": "4007",
    "21STCENMGM": "4",
    "NIFTY100EW": "29944",
    "011NSETEST": "14747",
    "091NSETEST": "14769",
    "111NSETEST": "14774",
    "GULFOILLUB": "4391",
    "JAGSNPHARM": "4410",
    "BHEL": "438",
    "IFCI": "1491",
    "RGL": "15129",
    "DELTACORP": "15044",
    "BRIGADE": "15184",
    "BLUEDART": "495",
    "QGOLDHALF": "15330",
    "NMDC": "15332",
    "DHANI": "15384",
    "EIHAHOTELS": "15399",
    "FMCGIETF": "5306",
    "UGROCAP": "5313",
    "WINDLAS": "5366",
    "IPCALAB": "1633",
    "ANDHRAPAP": "166",
    "KIRIINDUS": "16639",
    "SUNDROP": "1663",
    "ALMONDZ": "16719",
    "GVT&D": "16783",
    "ATGL": "6066",
    "ALKALI": "16959",
    "PARAGMILK": "17130",
    "BHAGCHEM": "6164",
    "EXCEL": "17376",
    "UTKARSHBNK": "17358",
    "CHAMBLFERT": "637",
    "BANKETF": "17419",
    "MANORG": "6422",
    "PREMEXPLN": "17397",
    "DNAMEDIA": "641",
    "OIL": "17438",
    "INCREDIBLE": "17507",
    "BSOFT": "6994",
    "ASALCBR": "17598",
    "SOMICONVEY": "17794",
    "MMTC": "17957",
    "GATECH": "17888",
    "MBLINFRA": "18029",
    "MICEL": "7169",
    "CONCORDBIO": "18060",
    "MEDPLUS": "7254",
    "UDAICEMENT": "7276",
    "JTLIND": "7287",
    "JIOFIN": "18143",
    "DCMSRIND": "7325",
    "NIFTY50ADD": "7451",
    "RAJESHEXPO": "7401",
    "AEROFLEX": "18268",
    "TREJHARA": "7518",
    "VPRPL": "18341",
    "SANDUMA": "18359",
    "EVIETF": "755881",
    "RISHABH": "18417",
    "JLHL": "18553",
    "EMSLIMITED": "18593",
    "SAMHI": "18614",
    "ENDURANCE": "18822",
    "PNBHOUSING": "18908",
    "KIRLOSIND": "19025",
    "HMVL": "19211",
    "TI": "19196",
    "LIQUIDETF": "1927",
    "BLISSGVS": "19265",
    "THEJO": "19279",
    "DALMIASUG": "781",
    "WEL": "7835",
    "HISARMETAL": "19322",
    "ROSSARI": "19410",
    "ALEMBICLTD": "79",
    "KSB": "1949",
    "HDFCGOLD": "19543",
    "MUFIN": "19783",
    "DALBHARAT": "8075",
    "HONASA": "19813",
    "GRWRHITECH": "7982",
    "TASTYBITE": "20092",
    "BEDMUTHA": "20196",
    "JWL": "20224",
    "IREDA": "20261",
    "MIDCAP": "8077",
    "IRFC": "2029",
    "SHANKARA": "20321",
    "RESPONIND": "20323",
    "RAMCOCEM": "2043",
    "BANKA": "822",
    "MUTHOOTMF": "20831",
    "MAHSEAMLES": "2088",
    "STOVEKRAFT": "2107",
    "MASTEK": "2124",
    "GMRP&UI": "8529",
    "STEELXIND": "21339",
    "LOWVOL1": "8632",
    "VRLLOG": "8696",
    "LIQUIDCASE": "21750",
    "MEDIASSIST": "21705",
    "SONAMLTD": "8806",
    "UMAEXPORTS": "8842",
    "GENCON": "2188",
    "VERANDA": "8890",
    "IOB": "9348",
    "ARVINDFASN": "9111",
    "PNCINFRA": "9385",
    "DPABHUSHAN": "936",
    "KPITTECH": "9683",
    "PGHL": "940",
    "SMARTLINK": "9889",
    "EUROTEXIND": "999",
    "UTIBANKETF": "22440",
    "EPACK": "22463",
    "NIFITETF": "22488",
    "PFS": "22602",
    "VSTL": "22730",
    "MTNL": "2294",
    "RKSWAMY": "23043",
    "KRYSTAL": "23175",
    "SINCLAIR": "23480",
    "AAKASH": "235",
    "MUTHOOTFIN": "23650",
    "LIQUIDBETF": "23915",
    "BBNPNBETF": "24117",
    "RETAIL": "24145",
    "DEEDEV": "24154",
    "NOCIL": "2442",
    "RUSHIL": "24595",
    "RELTD": "24596",
    "GROWWEV": "24798",
    "GSEC10ABSL": "24847",
    "LTF": "24948",
    "EBANKNIFTY": "25258",
    "BAJAJHFL": "25270",
    "GROWWLIQID": "25447",
    "SHAKTIPUMP": "25574",
    "INDOUS": "25503",
    "MOMENTUM50": "25606",
    "APLAPOLLO": "25780",
    "DBEIL": "25892",
    "SPECIALITY": "27107",
    "EIEL": "27213",
    "AUROPHARMA": "275",
    "PRSMJOHNSN": "2739",
    "INNOVANA": "27754",
    "RSWM": "2794",
    "RANASUG": "2837",
    "MAMATA": "28687",
    "RELIANCE": "2885",
    "CONSUMER": "28969",
    "TVSHLTD": "29008",
    "ZUARI": "29050",
    "NAVKARURB": "29146",
    "DENTA": "29256",
    "HEXT": "29666",
    "MSCIINDIA": "29758",
    "ORIENTELEC": "2972",
    "AONETOTAL": "29975",
    "WONDERLA": "3002",
    "AXISTECETF": "3010",
    "AXISVALUE": "30155",
    "SBIBPB": "30290",
    "SBIN": "3045",
    "SHANTIGEAR": "3078",
    "BAJAJHLDNG": "305",
    "COMPUSOFT": "31138",
    "MCX": "31181",
    "SHREDIGCEM": "3099",
    "SIMPLEXINF": "3162",
    "SIEMENS": "3150",
    "SHREYANIND": "3126",
    "SUNDRMFAST": "3345",
    "SUPERSPIN": "3357",
    "SPMLINFRA": "3321",
    "TFCILTD": "3466",
    "AGARIND": "3389",
    "TORNTPHARM": "3518",
    "TIL": "3484",
    "TATASTEEL": "3499",
    "VIPIND": "3703",
    "VOLTAS": "3718",
    "VENKEYS": "3757",
    "WHEELS": "3766",
    "PREMIERPOL": "3908",
    "HATSUN": "3892",
    "CAPLIPOINT": "3906",
    "TOKYOPLAST": "3837",
    "BIOFILCHEM": "4136",
    "BEPL": "419",
    "TNPL": "3980",
    "BHARATGEAR": "426",
    "DENORA": "4279",
    "5PAISA": "445",
    "AARTIDRUGS": "4481",
    "SELECTIPO": "30109",
    "EQUAL200": "30121",
    "3MINDIA": "474",
    "TCIFINANCE": "4771",
    "BANKINDIA": "4745",
    "SOTL": "3021",
    "SPECTRUM": "30455",
    "KPIGREEN": "5108",
    "JAICORPLTD": "5143",
    "SBIETFCON": "5168",
    "ICICIB22": "522",
    "ANTGRAPHIC": "5194",
    "DBSTOCKBRO": "31107",
    "ASAHIINDIA": "5378",
    "SUDARSCHEM": "3327",
    "ISGEC": "3329",
    "SUNFLAG": "3348",
    "NUVOCO": "5426",
    "BALKRISIND": "335",
    "SMLISUZU": "3387",
    "KOTHARIPRO": "5528",
    "THERMAX": "3475",
    "BALRAMCHIN": "341",
    "AXISBPSETF": "3530",
    "AXISBANK": "5900",
    "EIFFL": "6040",
    "AXISHCETF": "3608",
    "JUBLPHARMA": "3637",
    "MENONBE": "6961",
    "BBTC": "380",
    "ACCELYA": "7053",
    "MAHKTECH": "7074",
    "REMSONSIND": "3965",
    "NIACL": "399",
    "ARIHANTCAP": "3813",
    "BEL": "383",
    "SETFNIFBK": "7361",
    "BBETF0432": "7196",
    "SILINV": "4105",
    "CUBEXTUB": "4064",
    "EIMCOELECO": "4040",
    "MIDQ50ADD": "7456",
    "NRBBEARING": "7553",
    "WAAREERTL": "756038",
    "EQUAL50": "757142",
    "KEEPLEARN": "778",
    "MUKTAARTS": "8687",
    "STERTOOLS": "4299",
    "VISAKAIND": "4221",
    "BHARATFORG": "422",
    "VCL": "9020",
    "EIDPARRY": "916",
    "BANKBARODA": "4668",
    "DODLA": "4822",
    "GAIL": "4717",
    "SILVERCASE": "30274",
    "ICEMAKE": "489",
    "ENGINERSIN": "4907",
    "SHARDACROP": "4992",
    "PILITA": "30835",
    "GRINFRA": "5054",
    "INDUSINDBK": "5258",
    "GMDCLTD": "5204",
    "STYLAMIND": "5186",
    "BPCL": "526",
    "PDSL": "5264",
    "SERVOTECH": "5507",
    "MOL": "5394",
    "CARTRADE": "5407",
    "TATACHEM": "3405",
    "VSTIND": "3724",
    "RELINFRA": "553",
    "ANMOL": "3727",
    "ABMINTLLTD": "5549",
    "SINTERCOM": "381",
    "RITES": "3761",
    "WIPRO": "3787",
    "PVP": "4010",
    "BHARATRAS": "3834",
    "MASPTOP50": "5782",
    "BERGEPAINT": "404",
    "JAYAGROGN": "4041",
    "MARICO": "4067",
    "CANFINHOME": "583",
    "MIRZAINT": "4394",
    "SBCL": "4656",
    "RAMCOIND": "4587",
    "OSWALGREEN": "471",
    "HAPPSTMNDS": "48",
    "INDNIPPON": "4747",
    "LICNETFGSC": "6062",
    "3IINFOLTD": "6232",
    "LIQUIDIETF": "4838",
    "KIMS": "4847",
    "GULPOLY": "6286",
    "MADRASFERT": "4911",
    "TECH": "6462",
    "EQUAL50ADD": "6606",
    "MOLDTKPAC": "6713",
    "RPPL": "6913",
    "TARSONS": "6943",
    "MAHAPEXLTD": "5239",
    "HESTERBIO": "7048",
    "TEGA": "7105",
    "DEVYANI": "5373",
    "GRSE": "5475",
    "JAIPURKURT": "5516",
    "AMIORG": "5578",
    "ALPHA": "7412",
    "NILASPACES": "7411",
    "DATAPATTNS": "7358",
    "J&KBANK": "5633",
    "PARAS": "5911",
    "CIFL": "756015",
    "SOUTHBANK": "5948",
    "CARBORUNIV": "595",
    "COASTCORP": "6006",
    "ARTNIRMAN": "6156",
    "KRITI": "6417",
    "CAMLINFINE": "6216",
    "JINDALSTEL": "6733",
    "STEELCAS": "6803",
    "CIPLA": "694",
    "NETF": "7838",
    "GEPIL": "7862",
    "ANANDRATHI": "7145",
    "MOMOMENTUM": "8182",
    "BYKE": "7919",
    "SBIETFQLTY": "7218",
    "METROBRAND": "7242",
    "SBIETFIT": "740",
    "STAR": "7374",
    "MANAKALUCO": "7420",
    "MSUMI": "8596",
    "CHOICEIN": "8866",
    "USHAMART": "8840",
    "MON50EQUAL": "756900",
    "NILAINFRA": "9253",
    "KRITIKA": "9288",
    "CRISIL": "757",
    "AVANTIFEED": "7936",
    "ZYDUSLIFE": "7929",
    "SUBEXLTD": "967",
    "TRIDENT": "9685",
    "SILVER": "8003",
    "DCMSHRIRAM": "811",
    "AJANTPHARM": "8124",
    "SPENCERS": "8163",
    "PRITI": "8304",
    "ONEPOINT": "9939",
    "BLUESTARCO": "8311",
    "MIDCAPETF": "8413",
    "DHAMPURSUG": "857",
    "CAMPUS": "9362",
    "MAHESHWARI": "9576",
    "PARADEEP": "9741",
    "HAVELLS": "9819",
    "AXITA": "9902",
    "ORIENTCEM": "30089",
    "PRABHA": "30134",
    "SAMMAANCAP": "30125",
    "SDL26BEES": "3022",
    "SCHNEIDER": "31234",
    "ISHANCH": "30249",
    "GILT5YBEES": "3172",
    "BAJFINANCE": "317",
    "ANGELONE": "324",
    "SPIC": "3252",
    "SURYAROSNI": "3375",
    "SRF": "3273",
    "AJOONI": "3403",
    "CLSEL": "3482",
    "BBOX": "3435",
    "ORBTEXP": "31475",
    "MOKSH": "3586",
    "SKFINDIA": "3186",
    "SREEL": "31837",
    "VINDHYATEL": "3694",
    "CAMS": "342",
    "TATAELXSI": "3411",
    "MAHLOG": "385",
    "MAFANG": "3507",
    "BANARBEADS": "347",
    "TTKPRESTIG": "3546",
    "GREENPLY": "3987",
    "MOTHERSON": "4204",
    "KHADIM": "425",
    "UCAL": "3570",
    "NAM-INDIA": "357",
    "FINEORG": "3744",
    "HDFCLIFE": "467",
    "WEIZMANIND": "3748",
    "BATAINDIA": "371",
    "INDSWFTLAB": "4870",
    "FDC": "4898",
    "WENDT": "4235",
    "HINDCON": "4920",
    "IPL": "4934",
    "CREDITACC": "4421",
    "NEXT50IETF": "4529",
    "ALKYLAMINE": "4487",
    "DYNAMATECH": "4525",
    "RUCHINFRA": "4566",
    "HERITGFOOD": "4598",
    "SKMEGGPROD": "4732",
    "INDRAMEDCO": "4751",
    "BIRLACABLE": "477",
    "CONCOR": "4749",
    "PRITIKAUTO": "5292",
    "DCMFINSERV": "4775",
    "EPIGRAL": "5382",
    "GANESHBE": "5614",
    "SHANTI": "5650",
    "AXISCETF": "5732",
    "BANKNIFTY1": "5851",
    "ABSLAMC": "6018",
    "SGIL": "5114",
    "MAZDOCK": "509",
    "UTIAMC": "527",
    "SIKKO": "6218",
    "APTUS": "5435",
    "BRITANNIA": "547",
    "YUKEN": "5501",
    "FOCUS": "6836",
    "LINC": "6951",
    "ARCHIES": "5688",
    "AARTIIND": "7",
    "LIKHITHA": "578",
    "SEJALLTD": "7091",
    "SHALBY": "714",
    "RATEGAIN": "7177",
    "ABSLNN50ET": "7339",
    "COSMOFIRST": "742",
    "HPAL": "7376",
    "MONQ50": "7422",
    "FCL": "6198",
    "MEDICAMEQ": "6278",
    "ABREL": "625",
    "CONSUMIETF": "6446",
    "NYKAA": "6545",
    "GREENLAM": "6848",
    "LATENTVIEW": "6818",
    "MHLXMIRU": "7886",
    "RPSGVENT": "8119",
    "HEUBACHIND": "715",
    "ZODIAC": "7129",
    "SHRIRAMPPS": "7200",
    "JETFREIGHT": "7211",
    "MAPMYINDIA": "7227",
    "SUPRIYA": "7390",
    "COROMANDEL": "739",
    "MID150BEES": "8506",
    "TNIDETF": "8882",
    "PRIVISCL": "8825",
    "UMIYA-MRO": "8998",
    "ELDEHSG": "8953",
    "AGROPHOS": "9046",
    "TPLPLASTEH": "9219",
    "INDOAMIN": "9116",
    "ELECTCAST": "928",
    "ELGIEQUIP": "937",
    "RVNL": "9552",
    "SAKAR": "9539",
    "SHREERAMA": "7627",
    "DELHIVERY": "9599",
    "EMUDHRA": "9756",
    "IMAGICAA": "7672",
    "AUTOBEES": "7880",
    "NV20BEES": "9847",
    "MALLCOM": "7965",
    "CINEVISTA": "8024",
    "DCM": "805",
    "SHIVATEX": "804",
    "MAHLIFE": "8050",
    "CROWN": "8083",
    "KTKBANK": "8054",
    "DEVIT": "8146",
    "MANYAVAR": "8167",
    "SAGARDEEP": "8175",
    "FIBERWEB": "8159",
    "INDOBORAX": "8614",
    "RITCO": "8944",
    "UTISXN50": "9168",
    "STARTECK": "9305",
    "HARIOMPIPE": "8968",
    "RAINBOW": "9408",
    "MSTCLTD": "9356",
    "ESABINDIA": "955",
    "GOLDTECH": "9874",
    "NEOGEN": "9917",
    "VIKASLIFE": "9931",
    "MOLOWVOL": "8654",
    "GAEL": "8828",
    "UFO": "9039",
    "KHAICHEM": "896",
    "ANUP": "9014",
    "BALAJITELE": "9158",
    "STLTECH": "9309",
    "ESCORTS": "958",
    "LICI": "9480",
    "HIKAL": "9668",
    "RELCHEMQ": "9652",
    "METROPOLIS": "9581",
    "GOLD360": "30105",
    "KOHINOOR": "3009",
    "JINDALSAW": "3024",
    "NIF100IETF": "30392",
    "ESSENTIA": "30323",
    "MID150": "30469",
    "SCI": "3048",
    "VEDL": "3063",
    "INDOSTAR": "3088",
    "SHREECEM": "3103",
    "BARBEQUE": "3127",
    "TIINDIA": "312",
    "NBCC": "31415",
    "JTEKTINDIA": "3237",
    "STARPAPER": "3291",
    "DEEPINDS": "3292",
    "SUNPHARMA": "3351",
    "SUBROS": "3324",
    "TIRUMALCHM": "3496",
    "TATAPOWER": "3426",
    "SWARAJENG": "3384",
    "ADANIGREEN": "3563",
    "TNPETRO": "3509",
    "HEALTHIETF": "3626",
    "UNIVCABLES": "3607",
    "BASF": "368",
    "VARDMNPOLY": "3646",
    "VARROC": "3857",
    "ZUARIIND": "3827",
    "PARASPETRO": "3972",
    "BEML": "395",
    "HBSL": "4116",
    "REGENCERAM": "4130",
    "SUNDRMBRAK": "4179",
    "SOMATEX": "4201",
    "SHRIRAMFIN": "4306",
    "IFGLEXPOR": "436",
    "SIGIND": "4522",
    "BIRLACORPN": "480",
    "SNOWMAN": "4843",
    "ICICIBANK": "4963",
    "PHARMABEES": "4973",
    "IRCON": "4986",
    "ETERNAL": "5097",
    "CLEAN": "5049",
    "CIGNITITEC": "5142",
    "TATVA": "5162",
    "BPL": "530",
    "TRIGYN": "5428",
    "KRSNAA": "5359",
    "NGIL": "5401",
    "AAVAS": "5385",
    "ASTRAZEN": "5610",
    "CYIENT": "5748",
    "LAGNAM": "5865",
    "GRAPHITE": "592",
    "INTELLECT": "5926",
    "MONTECARLO": "5938",
    "DTIL": "6227",
    "CCHHL": "6385",
    "LAXMICOT": "6568",
    "FOODSIN": "6673",
    "FINOPB": "6579",
    "SIGACHI": "6663",
    "EXIDEIND": "676",
    "SBC": "6792",
    "CHOLAFIN": "685",
    "BAJAJHCARE": "6863",
    "DMCC": "6973",
    "GOCOLORS": "6964",
    "ASIANENE": "7030",
    "SETFNN50": "7353",
    "GLENMARK": "7406",
    "WOCKPHARMA": "7506",
    "GROWWMOM50": "756086",
    "UNITEDPOLY": "7560",
    "ATHERENERG": "757645",
    "DABUR": "772",
    "AUTOIETF": "7844",
    "MAKEINDIA": "7979",
    "DEEPAKFERT": "827",
    "SHAILY": "8727",
    "OBCL": "8797",
    "AVROIND": "8827",
    "TTML": "8954",
    "PIXTRANS": "9087",
    "TAJGVK": "9354",
    "KHANDSE": "9641",
    "SOUTHWEST": "9678",
    "CHEVIOT": "9879",
    "ABFRL": "30108",
    "SESHAPAPER": "3066",
    "RHIM": "31163",
    "BORORENEW": "3149",
    "JPOLYINVST": "31507",
    "THOMASCOOK": "3481",
    "BANARISUG": "350",
    "TITAN": "3506",
    "SOMANYCERA": "3880",
    "ZODIACLOTH": "3821",
    "AEGISLOG": "40",
    "AUSOMENT": "4037",
    "SENSEXIETF": "4378",
    "MPHASIS": "4503",
    "KOTHARIPET": "4594",
    "GANGAFORGE": "4957",
    "ADROITINFO": "4953",
    "ROLEXRINGS": "5279",
    "ALIVUS": "5265",
    "CHEMPLASTS": "5449",
    "ITDCEM": "5622",
    "VIJAYA": "5585",
    "SANSERA": "5751",
    "AIROLAM": "6068",
    "GOODLUCK": "6125",
    "CESC": "628",
    "NATIONALUM": "6364",
    "IGARASHI": "634",
    "LAMBODHARA": "6407",
    "TECHNOE": "6445",
    "GENUSPAPER": "6600",
    "GRPLTD": "6543",
    "SJS": "6643",
    "NIPPOBATRY": "6782",
    "SAPPHIRE": "6718",
    "PAYTM": "6705",
    "SMLT": "6843",
    "ICDSLTD": "6977",
    "HEMIPROP": "701",
    "SHRADHA": "708",
    "HCLTECH": "7229",
    "ZENTEC": "7508",
    "SILVER360": "756150",
    "NPST": "756324",
    "MONEXT50": "757167",
    "CMSINFO": "7603",
    "CGPOWER": "760",
    "BCONCEPTS": "7780",
    "INOXWIND": "7852",
    "SILVERBEES": "8080",
    "DCW": "817",
    "CHALET": "8546",
    "NLCINDIA": "8585",
    "VETO": "8652",
    "DRREDDY": "881",
    "E2E": "8937",
    "TIPSMUSIC": "9117",
    "EIHOTEL": "919",
    "CREATIVEYE": "9384",
    "PRUDENT": "9553",
    "POLYCAB": "9590",
    "VENUSPIPES": "9592",
    "MEDICO": "9667",
    "SPLPETRO": "9617",
}


interval_map = {
    "1 Month ": ("ONE_DAY", 30),
    "3 Months ": ("ONE_DAY", 90),
    "6 Months ": ("ONE_DAY", 180),
    "1 Year ": ("ONE_DAY", 365),
    "5 Days ": ("FIVE_MINUTE", 5)
}

all_tickers = get_nse_stock_list()
stock_choice = st.sidebar.selectbox("Select NSE Stock", all_tickers)
selected_label = st.sidebar.selectbox("Select Time Frame for Chart", list(interval_map.keys()))
interval_choice, days = interval_map[selected_label]

load_dotenv()

API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
PASSWORD = os.getenv("ANGEL_PASSWORD")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")

try:
    totp = pyotp.TOTP(TOTP_SECRET).now()
    obj = SmartConnect(api_key=API_KEY)
    session = obj.generateSession(CLIENT_ID, PASSWORD, totp)
except Exception as e:
    st.error(f"Login failed: {e}")
    st.stop()


st.markdown("""
    <div style='background-color:#111; color:#f39c12; padding:10px; border-radius:5px; margin-top:30px;
    overflow:hidden; white-space:nowrap; animation: scroll-left 12s linear infinite;'>
        ⚠️ Disclaimer: This dashboard is for informational purposes only and not a SEBI-registered advisory. Please consult a certified financial advisor before investing.
    </div>

    <style>
    @keyframes scroll-left {
    0%   {transform: translateX(100%);}
    100% {transform: translateX(-100%);}
    }
    </style>
    """, unsafe_allow_html=True)



# --- FETCH LIVE PRICE ---
try:
    token = symbol_map.get(stock_choice)
    live_data = obj.ltpData(exchange="NSE", tradingsymbol=stock_choice, symboltoken=token)
    ltp = round(live_data["data"]["ltp"], 2)
    st.metric(label=f"📈 Live Price of {stock_choice}", value=f"₹ {ltp}")
except Exception as e:
    st.warning(f"⚠️ Could not fetch live price. Reason: {e}")


# --- PAGE: Home ---
if page == "🏠 Home":
    st.title("📈 Market Overview Dashboard")


    # --- Index Prices from NSE India ---
    def fetch_all_indices():
        try:
            session = requests.Session()
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.nseindia.com"
            }
            session.get("https://www.nseindia.com", headers=headers)
            url = "https://www.nseindia.com/api/allIndices"
            r = session.get(url, headers=headers)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:
            print("Error fetching all indices:", e)
            return []

    def fetch_index_from_nse(index_symbol):
        try:
            data = fetch_all_indices()
            for index in data:
                if index_symbol.upper() in index['index'].upper():
                    return index['last'], index['percentChange']
            return "NA", "NA"
        except:
            return "NA", "NA"


    # --- Bing News Scraper ---
    def fetch_bing_news(query):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            url = f"https://www.bing.com/news/search?q={query.replace(' ', '+')}"
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, "html.parser")
            titles = soup.find_all("a", class_="title")
            return [t.text.strip() for t in titles[:5]]
        except:
            return []


    # --- Commodity from investing.com ---
    def get_commodity_price(name):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            url = f"https://www.investing.com/commodities/{name}"
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, "html.parser")
            price = soup.select_one("div[data-test='instrument-price-last'] span")
            return price.text.strip() if price else "NA"
        except:
            return "NA"


    # --- Fetch Index Data ---
    nifty, nifty_change = fetch_index_from_nse("NIFTY 50")
    niftyfinservice, niftyfinservice_change = fetch_index_from_nse("NIFTY FINANCIAL SERVICES")
    banknifty, banknifty_change = fetch_index_from_nse("NIFTY BANK")
    midcap100, midcap_change = fetch_index_from_nse("NIFTY MIDCAP 100")



    # 📊 Index Cards
    st.subheader("📊 Key Indices")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NIFTY 50", f"₹ {nifty}", f"{nifty_change}%")
    col2.metric("BANK NIFTY", f"₹ {banknifty}", f"{banknifty_change}%")
    col3.metric("FINNIFTY", f"₹ {niftyfinservice}", f"{niftyfinservice_change}%")
    col4.metric("MIDCAP 100", f"₹ {midcap100}", f"{midcap_change}%")


    # ──────────────────────────────────────────────
    # 📥 Scraping Functions for Each Corporate Action
    # ──────────────────────────────────────────────

    # --- Animated Title ---
    st.markdown("""
    <style>
    @keyframes popZoom {
    0% {transform: scale(0.95); opacity: 0;}
    100% {transform: scale(1); opacity: 1;}
    }
    .animated-heading {
        animation: popZoom 1s ease-out forwards;
        font-size: 36px;
        text-align: center;
        color: #ffffff;
        margin-bottom: 20px;
        font-weight: bold;
    }
    </style>
    <h2 class="animated-heading">📅 Market Event Highlights</h2>
    """, unsafe_allow_html=True)

    # --- Lottie Loader ---
    def load_lottie_url(url):
        try:
            r = requests.get(url)
            return r.json() if r.status_code == 200 else None
        except:
            return None

    # --- Bing News Fetcher ---
    def bing_news_search(query, count=4):
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.bing.com/news/search?q={query.replace(' ', '+')}+site:moneycontrol.com&FORM=HDRSC6"
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for item in soup.select("a.title")[:count]:
            results.append({
                "title": item.text.strip(),
                "url": item["href"]
            })
        return results

    # --- Load Working Animations ---
    dividend_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
    split_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_5ngs2ksb.json")
    meeting_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_ol7a7z.json")
    rights_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")

    # --- Define Categories ---
    topics = {
        "📦 Dividends": {"query": "stock dividend announcement India", "anim": dividend_lottie},
        "🔀 Stock Splits": {"query": "stock split announced India", "anim": split_lottie},
        "📅 Board Meetings": {"query": "stock market board meeting agenda India", "anim": meeting_lottie},
        "🎫 Rights Issues": {"query": "stock market rights issue declared India", "anim": rights_lottie},
    }

    # --- Render News in 4 Columns with Vertical Scroll ---
    cols = st.columns(4)
    for i, (label, details) in enumerate(topics.items()):
        with cols[i]:
            st.markdown(f"<div style='text-align:center; font-size:18px; font-weight:bold; margin-bottom:10px;'>{label}</div>", unsafe_allow_html=True)

            if details["anim"]:
                st_lottie(details["anim"], height=100, key=label)

            news = bing_news_search(details["query"])

            if news:
                scroll_css = f"""
                <style>
                .scroll-box-{i} {{
                    height: 200px;
                    overflow: hidden;
                    position: relative;
                }}
                .scroll-content-{i} {{
                    display: block;
                    position: absolute;
                    top: 100%;
                    animation: scroll-up-{i} 15s linear infinite;
                }}
                .scroll-content-{i} div {{
                    margin-bottom: 18px;
                    line-height: 1.4;
                }}
                @keyframes scroll-up-{i} {{
                    0%   {{ top: 100%; }}
                    100% {{ top: -100%; }}
                }}
                </style>
                """
                st.markdown(scroll_css, unsafe_allow_html=True)
                st.markdown(f"<div class='scroll-box-{i}'><div class='scroll-content-{i}'>", unsafe_allow_html=True)
                for item in news:
                    st.markdown(
                        f"<div><a href='{item['url']}' target='_blank' style='text-decoration: none; color: #4ADE80;'>🔗 {item['title']}</a></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color: gray;'>No recent news found.</div>", unsafe_allow_html=True)

    # --- Footer ---
    st.markdown(
        "<br><div style='text-align:center; font-size:13px; color:gray;'>📱 Powered by Bing News • Source: Moneycontrol</div>",
        unsafe_allow_html=True
    )
  


    # 📂 Market Sentiment Summary
    st.subheader("📂 Market Sentiment Summary")
    try:
        change = float(nifty_change)
        if change > 0:
            st.success("📈 Market is Bullish")
        elif change < 0:
            st.error("📉 Market is Bearish")
        else:
            st.info("📊 Market is Flat")
    except:
        st.warning("Could not determine sentiment")

    # 🗞️ Top Headlines
    st.subheader("🗞️ Top Headlines")
    headlines = fetch_bing_news("Nifty 50")
    if headlines:
        for i, news in enumerate(headlines, 1):
            st.markdown(f"**{i}. {news}**")
    else:
        st.info("No headlines available.")

    # 📈 Top Gainers & Losers
    st.subheader("📈 Top Gainers & Losers")
    g1, g2 = st.columns(2)
    g1.markdown("**Top Gainers**")
    for g in fetch_bing_news("nifty top gainers"):
        g1.write(f"- {g}")
    g2.markdown("**Top Losers**")
    for l in fetch_bing_news("nifty top losers"):
        g2.write(f"- {l}")

    # 📤 Portfolio Upload & Risk Analyzer
    st.subheader("📤 Portfolio Risk Analyzer")
    uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            pf_df = pd.read_csv(uploaded_file)
            st.write("Your Portfolio:", pf_df)
            risk_score = pf_df["Weight"].std() if "Weight" in pf_df.columns else "N/A"
            st.success(f"📊 Risk Score: {round(risk_score, 2)}")
        except:
            st.error("❌ Failed to read the uploaded file.")

    # 🔎 Watchlist-based News
    st.subheader("🔎 News Based on Your Watchlist")
    if "watchlist" in st.session_state and st.session_state.watchlist:
        for stock in st.session_state.watchlist[:3]:
            st.markdown(f"**📌 {stock}**")
            for item in fetch_bing_news(stock):
                st.write(f"- {item}")
    else:
        st.info("Add stocks to your Watchlist to get personalized news.")


# --- PAGE: Fundamentals ---
elif page == "🔍 Fundamentals":
    st.title("📊 Key Financial Ratios")

    # Screener-compatible company slugs
    screener_slug_map = {
        "RELIANCE": "RELIANCE",
        "TCS": "TCS",
        "INFY": "INFOSYS",
        "SBIN": "SBIN",
        "HDFCBANK": "HDFCBANK",
        "ITC": "ITC",
        "AXISBANK": "AXISBANK",
        "BAJFINANCE": "BAJFINANCE",
        "HCLTECH": "HCLTECH",
        "LT": "LT",
        "WIPRO": "WIPRO",
        "ICICIBANK": "ICICIBANK",
        "KOTAKBANK": "KOTAKBANK",
        "ONGC": "ONGC",
        "ADANIENT": "ADANIENT",
        "POWERGRID": "POWERGRID",
        "NTPC": "NTPC",
        "COALINDIA": "COALINDIA",
        "HINDUNILVR": "HINDUNILVR",
        "ULTRACEMCO": "ULTRACEMCO"
    }

    # ✅ CORRECTLY INDENTED FUNCTION
    def fetch_fundamentals(ticker):
        try:
            slug = screener_slug_map.get(ticker, ticker)
            url = f"https://www.screener.in/company/{slug}/"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                st.warning("⚠️ Screener is currently not reachable.")
                return pd.DataFrame()

            soup = BeautifulSoup(response.text, "html.parser")
            data = []

            for item in soup.select("ul#top-ratios li"):
                label_elem = item.select_one("span.name")
                value_elem = item.select_one("span.value")
                if label_elem and value_elem:
                    label = label_elem.text.strip()
                    value = value_elem.text.strip()
                    data.append({"Metric": label, "Value": value})

            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            return pd.DataFrame()

    # Fetch data
    df = fetch_fundamentals(stock_choice)

    if not df.empty:
        num_cols = 2
        total_metrics = len(df)
        rows = total_metrics // num_cols + int(total_metrics % num_cols != 0)

        st.markdown("### 🧮 Organized Ratio Grid")
        for row in range(rows):
            cols = st.columns(num_cols)
            for i in range(num_cols):
                idx = row * num_cols + i
                if idx < total_metrics:
                    metric = df.iloc[idx]["Metric"]
                    value = df.iloc[idx]["Value"]
                    with cols[i]:
                        st.markdown(
                            f"""
                            <div style='background-color:#111111; padding:20px; border-radius:12px;
                            border:1px solid #333333; text-align:center; margin-bottom:10px;'>
                                <h5 style='color:#aaaaaa; margin-bottom:8px;'>{metric}</h5>
                                <h3 style='color:#29b6f6;'>{value}</h3>
                            </div>
                            """, unsafe_allow_html=True
                        )
    else:
        st.warning("⚠️ Could not fetch fundamentals for this stock.")

# --- PAGE: Charts & Indicators ---
elif page == "📉 Charts & Indicators":
    try:
        st.title("📉 Technical Charts & AI Signals")

        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=days)

        candle_data = obj.getCandleData({
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval_choice,
            "fromdate": start_time.strftime("%Y-%m-%d %H:%M"),
            "todate": end_time.strftime("%Y-%m-%d %H:%M")
        })

        df = pd.DataFrame(candle_data["data"], columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

        # Indicators
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()

        # AI-style signal generation
        df["Signal"] = "Hold"
        for i in range(1, len(df)):
            if df["EMA_10"].iloc[i] > df["SMA_20"].iloc[i] and df["EMA_10"].iloc[i - 1] <= df["SMA_20"].iloc[i - 1] and df["RSI"].iloc[i] < 70:
                df.at[df.index[i], "Signal"] = "Buy"
            elif df["EMA_10"].iloc[i] < df["SMA_20"].iloc[i] and df["EMA_10"].iloc[i - 1] >= df["SMA_20"].iloc[i - 1] and df["RSI"].iloc[i] > 30:
                df.at[df.index[i], "Signal"] = "Sell"

        # Buy/Sell points for plotting
        buy_signals = df[df["Signal"] == "Buy"]
        sell_signals = df[df["Signal"] == "Sell"]

        # --- Chart with OHLC + Volume + Signals ---
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        ))

        # EMA/SMA overlays
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_10"], mode="lines", name="EMA 10", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20", line=dict(color="blue")))

        # --- Buy markers with text ---
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals["Close"],
            mode="markers+text",
            name="Buy",
            text=["Buy"] * len(buy_signals),
            textposition="top center",
            marker=dict(color="lime", size=10, symbol="triangle-up")
        ))

        # --- Sell markers with text ---
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals["Close"],
            mode="markers+text",
            name="Sell",
            text=["Sell"] * len(sell_signals),
            textposition="bottom center",
            marker=dict(color="red", size=10, symbol="triangle-down")
        ))

        # Volume bars
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker=dict(color="lightblue"), yaxis="y2"))

        # Layout adjustments for volume axis
        fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_dark",
            title=f"{stock_choice} - Chart with AI Signals"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- RSI + MACD ---
        st.subheader("📈 RSI & MACD Indicators")
        st.line_chart(df[["RSI"]])
        st.line_chart(df[["MACD", "MACD_Signal"]])

    except Exception as e:
        st.error(f"Error: {e}")
elif page == "🤖 AI Prediction":
    try:
        st.title("🤖 AI-Based Advanced Signal Engine v2.0")

        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=days)

        candle_data = obj.getCandleData({
            "exchange": "NSE",  
            "symboltoken": token,
            "interval": interval_choice,
            "fromdate": start_time.strftime("%Y-%m-%d %H:%M"),
            "todate": end_time.strftime("%Y-%m-%d %H:%M")
        })

        df = pd.DataFrame(candle_data["data"], columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

        # 🛡️ Check if we have at least 20 rows to proceed
        if df.shape[0] < 20:
            st.warning("⚠️ Not enough data to generate prediction. Please select a longer time frame.")
        else:
            # --- Indicators ---
            df["EMA_20"] = df["Close"].ewm(span=20).mean()
            df["EMA_50"] = df["Close"].ewm(span=50).mean()
            df["EMA_200"] = df["Close"].ewm(span=200).mean()
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            macd = ta.trend.MACD(df["Close"])
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
            df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
            df["PSAR"] = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"]).psar()
            df["Bollinger High"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
            df["Bollinger Low"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()

            latest = df.iloc[-1]
            prev = df.iloc[-2]
            signal = "Hold"
            score = 0
            reasons = []

            # --- Signal Decision Logic (same as before) ---
            if latest["EMA_20"] > latest["EMA_50"] > latest["EMA_200"]:
                score += 2
                reasons.append("✅ Strong bullish EMA alignment")
            elif latest["EMA_20"] < latest["EMA_50"] < latest["EMA_200"]:
                score -= 2
                reasons.append("⛔ Bearish EMA alignment")

            if latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
                score += 2
                reasons.append("✅ MACD Bullish Crossover")
            elif latest["MACD"] < latest["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
                score -= 2
                reasons.append("⛔ MACD Bearish Crossover")

            if latest["RSI"] < 30:
                score += 1
                reasons.append("✅ Oversold RSI (<30)")
            elif latest["RSI"] > 70:
                score -= 1
                reasons.append("⛔ Overbought RSI (>70)")

            if latest["Close"] > latest["PSAR"]:
                score += 1
                reasons.append("✅ PSAR indicates uptrend")
            else:
                score -= 1
                reasons.append("⛔ PSAR indicates downtrend")

            if latest["ADX"] > 25:
                score += 1
                reasons.append("✅ Strong trend confirmed by ADX")
            else:
                reasons.append("ℹ️ Weak trend (ADX < 25)")

            avg_volume = df["Volume"].rolling(20).mean().iloc[-1]
            if latest["Volume"] > avg_volume:
                score += 1
                reasons.append("✅ Volume supports price movement")
            else:
                reasons.append("⚠️ Weak volume")

            # Final Signal Decision
            if score >= 4:
                signal = "Buy"
            elif score <= -3:
                signal = "Sell"
            else:
                signal = "Hold"

            recent_high = df["High"].rolling(10).max().iloc[-1]
            recent_low = df["Low"].rolling(10).min().iloc[-1]
            target = round(recent_high * 1.02, 2)
            stop_loss = round(recent_low, 2)
            reward = round(target - latest["Close"], 2)
            risk = round(latest["Close"] - stop_loss, 2)
            rr_ratio = round(reward / risk, 2) if risk > 0 else "∞"
            confidence = f"{min(abs(score) * 20, 95)}%"

            # Display Results
            st.metric("📍 Final Signal", signal)
            st.markdown(f"**🎯 Confidence Level:** `{confidence}`")
            for r in reasons:
                st.markdown(f"- {r}")

            if signal == "Buy":
                st.success(f"🎯 Target: ₹{target}")
                st.warning(f"🛡️ Stop Loss: ₹{stop_loss}")
            elif signal == "Sell":
                st.error(f"📉 Downside Alert: Below ₹{stop_loss}")
                st.warning(f"🛡️ Resistance: ₹{target}")
            else:
                st.info("⚖️ Market indecisive. Wait for confirmation.")

            # Extra Metrics
            st.subheader("📈 Technical Stats")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔄 ATR", round(latest["ATR"], 2))
            col2.metric("📊 ADX", round(latest["ADX"], 2))
            col3.metric("⚖️ Risk/Reward", rr_ratio)

            # Chart
            st.subheader("📉 Price Trend Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="white")))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], name="EMA 50", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_200"], name="EMA 200", line=dict(color="lime")))
            fig.update_layout(title=f"{stock_choice} | Advanced AI Prediction", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")


# News Sentiment page

elif page == "📰 News Sentiment":
    st.title("📰 News Sentiment")
    st.subheader("🔍 Latest Headlines")
    query = st.text_input("Search news for:", value=stock_choice)

    # --- Inline CSS ---
    st.markdown("""
        <style>
        .news-card {
            background-color: #1e1e1e;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.05);
            transition: transform 0.2s ease;
            height: 100%;
        }
        .news-card:hover {
            transform: scale(1.02);
        }
        .news-img {
            width: 100%;
            height: 160px;
            object-fit: cover;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .news-title {
            color: #29b6f6 !important;
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.4;
            text-decoration: none;
        }
        .news-title:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

    def fetch_news_sentiment(query):
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            url = f"https://www.bing.com/news/search?q={query}+stock&form=QBNH&sp=-1&ghc=1&filters=ex1%3a%22ez5_{start_date}_{end_date}%22"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.find_all("a", class_="title", href=True)

            news_data = []
            for a in articles[:6]:
                title = a.text.strip()
                link = a["href"]

                # Try to get image
                img_url = "https://i.imgur.com/EO8a4Tz.png"
                parent = a.find_parent("div")
                if parent:
                    img_tag = parent.find("img")
                    if img_tag and "src" in img_tag.attrs:
                        img_url = img_tag["src"]

                # Sentiment detection
                title_lower = title.lower()
                if any(x in title_lower for x in ["profit rises", "record high", "beats estimates", "surges", "jumps", "growth"]):
                    sentiment = "📈 Bullish"
                    tag_color = "green"
                elif any(x in title_lower for x in ["falls", "layoff", "misses", "dips", "plunge", "cut", "drops", "decline", "loss"]):
                    sentiment = "📉 Bearish"
                    tag_color = "red"
                else:
                    sentiment = "❕ Neutral"
                    tag_color = "gray"

                # Category Tag
                if "result" in title_lower:
                    category = "📊 Results"
                elif "alert" in title_lower:
                    category = "🚨 Alert"
                elif "price" in title_lower or "share" in title_lower:
                    category = "💹 Price"
                else:
                    category = "📰 General"

                news_data.append((title, link, img_url, sentiment, tag_color, category))
            return news_data
        except Exception as e:
            print("News fetch error:", e)
            return []

    news_items = fetch_news_sentiment(query)

    if news_items:
        cols = st.columns(3)
        for i, (title, link, img, sentiment, color, category) in enumerate(news_items):
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="news-card">
                        <img src="{img}" class="news-img" alt="thumbnail">
                        <a href="{link}" target="_blank" class="news-title">{title}</a>
                        <div style="margin-top:8px;">
                            <span style="background-color:{color}; color:white; padding:4px 8px; border-radius:5px; font-size: 0.75rem;">{sentiment}</span>
                            <span style="background-color:#444; color:white; padding:4px 8px; border-radius:5px; font-size: 0.75rem; margin-left:5px;">{category}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No recent news found.")


# --- PAGE: Watchlist ---
elif page == "⭐ Watchlist":
    st.title("⭐ Your Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    if st.button("Add to Watchlist"):
        if stock_choice not in st.session_state.watchlist:
            st.session_state.watchlist.append(stock_choice)
    st.write("Watchlist:", st.session_state.watchlist)

# --- PAGE: Advanced Analysis ---
elif page == "📊 Advanced Analysis":
    st.title("📊 Advanced Market Analysis")

    try:
        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=days)

        candle_data = obj.getCandleData({
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval_choice,
            "fromdate": start_time.strftime("%Y-%m-%d %H:%M"),
            "todate": end_time.strftime("%Y-%m-%d %H:%M")
        })

        df = pd.DataFrame(candle_data["data"], columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

        # --- Advanced Indicators ---
        df["ATR"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
        df["EMA_20"] = df["Close"].ewm(span=20).mean()
        df["Bollinger High"] = df["Close"].rolling(window=20).mean() + 2 * df["Close"].rolling(window=20).std()
        df["Bollinger Low"] = df["Close"].rolling(window=20).mean() - 2 * df["Close"].rolling(window=20).std()

        # --- Swing Zones ---
        recent_high = df["High"].rolling(10).max().iloc[-1]
        recent_low = df["Low"].rolling(10).min().iloc[-1]

        # --- Price Gaps ---
        gap = df["Open"].iloc[-1] - df["Close"].iloc[-2]
        gap_msg = "⬆️ Up Gap" if gap > 0 else "⬇️ Down Gap" if gap < 0 else "No Gap"

        # --- Volatility Summary ---
        st.subheader("📈 Price Volatility & Ranges")
        st.metric("🔄 ATR (14)", round(df["ATR"].iloc[-1], 2))
        st.metric("🔍 Bollinger Band Width", round(df["Bollinger High"].iloc[-1] - df["Bollinger Low"].iloc[-1], 2))

        # --- Price Action Summary ---
        st.subheader("📍 Swing Analysis")
        st.write(f"🔺 Recent Resistance: ₹ {round(recent_high, 2)}")
        st.write(f"🔻 Recent Support: ₹ {round(recent_low, 2)}")
        st.write(f"📊 Gap Detected: {gap_msg} ({round(gap, 2)})")

        # --- Chart with Bands ---
        st.subheader("📊 Technical View (Bollinger Bands)")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger High"], name="Upper Band", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger Low"], name="Lower Band", line=dict(color="red")))
        st.plotly_chart(fig, use_container_width=True)

        # --- Summary AI Signal ---
        st.subheader("🧠 AI Summary & Recommendation")
        if gap > 0 and df["Close"].iloc[-1] > recent_high:
            st.success("🚀 Strong Bullish Breakout Detected – Buy Signal")
        elif gap < 0 and df["Close"].iloc[-1] < recent_low:
            st.error("⚠️ Bearish Breakdown – Caution Advised")
        else:
            st.info("🔄 Sideways/Neutral Trend – Wait for confirmation")

             # --- 📊 Advanced Add-ons: LSTM Forecast + News Sentiment + Backtesting ---

        

        st.markdown("## 🔁 AI Enhancements: LSTM | Sentiment | Backtesting")

        # --- 1️⃣ LSTM Price Prediction ---
        try:
            st.subheader("🔮 LSTM Forecast (Next Day)")
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[["Close"]])
            look_back = 60

            X, y = [], []
            for i in range(look_back, len(scaled_data)):
                X.append(scaled_data[i - look_back:i, 0])
                y.append(scaled_data[i, 0])
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(X, y, epochs=10, batch_size=8, verbose=0)

            last_sequence = scaled_data[-look_back:]
            last_sequence = np.reshape(last_sequence, (1, look_back, 1))
            next_day_scaled = model.predict(last_sequence)
            next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

            st.metric("📈 Predicted Price (Tomorrow)", f"₹ {round(next_day_price, 2)}")
        except Exception as e:
            st.warning(f"LSTM prediction error: {e}")

        # --- 2️⃣ News Sentiment Analysis (Bing + FinBERT Simplified) ---
        try:
            st.subheader("📰 News Sentiment (Bing Headlines)")
            def fetch_headlines(stock):
                headers = {"User-Agent": "Mozilla/5.0"}
                url = f"https://www.bing.com/news/search?q={stock}+stock&FORM=HDRSC6"
                r = requests.get(url, headers=headers)
                soup = BeautifulSoup(r.text, "html.parser")
                links = soup.find_all("a", class_="title")
                return [link.text.strip() for link in links[:5]]

            headlines = fetch_headlines(stock_choice)
            positive, negative = 0, 0

            for title in headlines:
                title_lower = title.lower()
                if any(x in title_lower for x in ["surge", "beat", "gain", "profit", "up", "grow"]):
                    positive += 1
                elif any(x in title_lower for x in ["fall", "loss", "drop", "down", "miss", "cut"]):
                    negative += 1

            total = positive + negative if (positive + negative) > 0 else 1
            score = round((positive - negative) / total * 100, 2)

            sentiment = "📈 Bullish" if score > 20 else "📉 Bearish" if score < -20 else "⚖️ Neutral"
            st.metric("🧠 Sentiment Score", f"{score}%", delta=sentiment)

            for i, h in enumerate(headlines, 1):
                st.markdown(f"**{i}.** {h}")

        except Exception as e:
            st.warning(f"Sentiment scraping failed: {e}")

        # --- 3️⃣ Backtesting with backtesting.py ---
        try:
            st.subheader("📉 Strategy Backtesting")

            class MAStrategy(Strategy):
                def init(self):
                    self.ema20 = self.I(lambda x: x.ewm(span=20).mean(), self.data.Close)
                    self.ema50 = self.I(lambda x: x.ewm(span=50).mean(), self.data.Close)

                def next(self):
                    if self.ema20[-1] > self.ema50[-1] and self.ema20[-2] <= self.ema50[-2]:
                        self.buy()
                    elif self.ema20[-1] < self.ema50[-1] and self.ema20[-2] >= self.ema50[-2]:
                        self.sell()

            bt = Backtest(df, MAStrategy, cash=100000, commission=.002, exclusive_orders=True)
            stats = bt.run()
            st.write("📊 Backtest Summary", stats[['Return [%]', 'Win Rate [%]', '# Trades']])
            st.plotly_chart(bt.plot(open_browser=False), use_container_width=True)
        except Exception as e:
            st.warning(f"Backtesting error: {e}")


        # --- PROPHET FORECAST SECTION ---
        st.subheader("📆 30-Day Price Forecast (Prophet AI)")

        try:
            from prophet import Prophet

            # Prepare data for Prophet
            prophet_df = df.reset_index()[["Datetime", "Close"]].rename(columns={"Datetime": "ds", "Close": "y"}).dropna()
            prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)  # 🚫 Remove timezone info


            if len(prophet_df) >= 60:  # Ensure enough data
                model = Prophet(daily_seasonality=True)
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Predicted Price", line=dict(color="orange")))
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper Bound", line=dict(color="lightgreen"), opacity=0.3))
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower Bound", line=dict(color="red"), opacity=0.3))
                fig2.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual", line=dict(color="white")))

                fig2.update_layout(
                    title=f"{stock_choice} – 30-Day Price Forecast (Prophet)",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Tomorrow forecast
                tomorrow_index = len(prophet_df) + 1
                if tomorrow_index < len(forecast):
                    tomorrow_pred = forecast.iloc[tomorrow_index]
                    st.metric("🔮 Tomorrow's Forecasted Price", f"₹ {round(tomorrow_pred['yhat'], 2)}")
                else:
                    st.warning("Insufficient forecast data to display tomorrow's price.")
            else:
                st.warning("📉 Not enough historical data to run 30-day Prophet forecast. Please use a longer time frame (e.g., 3M or 6M).")

        except Exception as e:
            st.error(f"Prophet Forecast Error: {e}")

    except Exception as e:
        st.error(f"Advanced Analysis Error: {e}")


