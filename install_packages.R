install.packages("renv")
renv::init()
renv::snapshot()  # Guarda estado de paquetes

# VECM y cointegración
install.packages("urca")        # Unit root tests, Johansen
install.packages("vars")        # VAR/VECM
install.packages("tsDyn")       # Threshold cointegration

# Diagnósticos
install.packages("lmtest")      # Breusch-Godfrey, etc.
install.packages("tseries")     # Jarque-Bera, KPSS

# Hidden Markov
install.packages("depmixS4")    # HMM con covariables

# Visualización
install.packages("ggplot2")
install.packages("gridExtra")

# Utilities
install.packages("readr")
install.packages("dplyr")
install.packages("tidyr")


