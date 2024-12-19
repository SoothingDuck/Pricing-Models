from markov import glo_min, g_pi_t, g_t, s_p, simulatedata

## Functions
def fit_MSM(data, kbar, niter=1, temperature=1.0, stepsize=1.0):
    parameters, LL, niter, output = glo_min(
        kbar,
        data,
        niter = niter,
        temperature = temperature,
        stepsize = stepsize
    )

    b = parameters[0]
    m0 = parameters[1]
    gamma_kbar = parameters[2]
    sigma = parameters[3]       # sigma simulation

    return(b, m0, gamma_kbar, sigma)

def predict_vol(data, kbar, b, m0, gamma_kbar, sigma, h = None):
    g_m = s_p(kbar, m0)
    theta_in = [b, gamma_kbar, sigma]
    smoothed_p = g_pi_t(m0, kbar, data, theta_in)
    A = g_t(kbar, b, gamma_kbar)

    if h is None:
        vol = np.sqrt((sigma**2)*np.dot(smoothed_p, g_m**2))
    else:
        next_A = A
        for i in range(h):
            next_A = np.dot(next_A, next_A)
        next_smoothed_p = np.dot(smoothed_p, next_A)
        vol = np.sqrt((sigma**2)*np.dot(next_smoothed_p, g_m**2))

    return(vol)


def get_tickers_slickcharts(ndx):
    import requests
    from bs4 import BeautifulSoup

    url = f'https://www.slickcharts.com/{ndx}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, features="lxml")

    table = soup.find('table')  # Find the table
    header = []                 # Init header list
    rows = []                   # Init rows
    # Iterate through all the table rows
    # First row is the header
    for i, row in enumerate(table.find_all('tr')):
        if i == 0:
            header = [el.text.strip() for el in row.find_all('th')]
        else:
            rows.append([el.text.strip() for el in row.find_all('td')])
    # Copy the rows and header into the dataframe
    tickers = pd.DataFrame(rows, columns=header)
    return tickers.loc[~pd.isnull(tickers.Symbol),:].reset_index()


## Maintenant
# Set kbar
kbar = 5

b = 6
m0 = 1.6
gamma_kbar = 0.8
sig = 2/np.sqrt(252)
T = 7087
E = np.rint(0.6*T).astype(int)
dat1 = simulatedata(b,m0,gamma_kbar,sig,kbar,T)

dat1E = dat1[0:E,]
dat1F = dat1[E:,]
data = dat1E               # Simulated dta

b, m0, gamma_kbar, sigma = fit_MSM(data, kbar)

# predict
predict_vol(data, kbar, b, m0, gamma_kbar, sigma)*np.random.normal()

# sur le vrai
import yfinance as yf

nasdaq100 = get_tickers_slickcharts("nasdaq100") 
