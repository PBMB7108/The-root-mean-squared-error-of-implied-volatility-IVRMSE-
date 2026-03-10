import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# ==========================================
# Black–Scholes Call Price
# ==========================================

def bs_call(S,K,r,T,sigma):

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


# ==========================================
# Implied Volatility
# ==========================================

def implied_volatility(price,S,K,r,T):

    try:

        f = lambda sigma: bs_call(S,K,r,T,sigma) - price

        return brentq(f,1e-6,5)

    except:

        return np.nan


# ==========================================
# Load Data
# ==========================================

hz = pd.read_excel("Resumen_HZ.xlsx",sheet_name="Resultados")

heston = pd.read_excel(
    "Resumen_Heston_Tradicional_COMAS.xlsx",
    sheet_name="Resultados"
)

conf = pd.read_excel(
    "Optimizaciones_Internacionales.xlsx",
    sheet_name="Resumen"
)

# ==========================================
# Merge Heston + HeZhu
# ==========================================

data = hz.merge(heston,on=["Empresa","Hoja"])

# ==========================================
# Convert units
# ==========================================

data["T_years"] = data["Vencimiento (T)"]/365
data["r_annual"] = data["r"]*365

# ==========================================
# Market Implied Volatility
# ==========================================

data["IV_market"] = data.apply(
    lambda x: implied_volatility(
        x["Precio de Mercado"],
        x["S0"],
        x["Strike"],
        x["r_annual"],
        x["T_years"]
    ),
axis=1)

# ==========================================
# Heston Implied Volatility
# ==========================================

data["IV_Heston"] = data.apply(
    lambda x: implied_volatility(
        x["Precio Opción Heston (B4)"],
        x["S0"],
        x["Strike"],
        x["r_annual"],
        x["T_years"]
    ),
axis=1)

# ==========================================
# He-Zhu Implied Volatility
# ==========================================

data["IV_HZ"] = data.apply(
    lambda x: implied_volatility(
        x["Precio Opción HZ"],
        x["S0"],
        x["Strike"],
        x["r_annual"],
        x["T_years"]
    ),
axis=1)

# ==========================================
# Conformable Model
# ==========================================

conf = conf.merge(heston[["Empresa","Hoja","S0","r"]],
                  on=["Empresa","Hoja"],
                  how="left")

conf["T_years"] = conf["T"]/365
conf["r_annual"] = conf["r"]*365

conf["IV_market"] = conf.apply(
    lambda x: implied_volatility(
        x["Precio Mercado"],
        x["S0"],
        x["Strike"],
        x["r_annual"],
        x["T_years"]
    ),
axis=1)

conf["IV_conf"] = conf.apply(
    lambda x: implied_volatility(
        x["Precio Ajustado"],
        x["S0"],
        x["Strike"],
        x["r_annual"],
        x["T_years"]
    ),
axis=1)

# ==========================================
# Compute IVRMSE
# ==========================================

IVRMSE_Heston = np.sqrt(
    np.nanmean((data["IV_Heston"] - data["IV_market"])**2)
)

IVRMSE_HZ = np.sqrt(
    np.nanmean((data["IV_HZ"] - data["IV_market"])**2)
)

IVRMSE_Conf = np.sqrt(
    np.nanmean((conf["IV_conf"] - conf["IV_market"])**2)
)

print("IVRMSE Heston:", IVRMSE_Heston)
print("IVRMSE He-Zhu:", IVRMSE_HZ)
print("IVRMSE Conformable:", IVRMSE_Conf)