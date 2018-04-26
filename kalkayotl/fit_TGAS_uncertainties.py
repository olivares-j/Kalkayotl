import numpy  as np
import pandas as pn
import scipy.stats as st
import scipy.optimize as optimization

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

u = np.array(pn.read_csv("TGAS_parallax_uncertainty.csv"))

param = st.chi2.fit(u)
print param

x = np.arange(min(u),max(u),(max(u)-min(u))/100)

pdf = PdfPages(filename="TGAS_parallax_uncertainty.pdf")
plt.hist(u,density=True,label="TGAS parallax uncertainties")
pdf_fitted = st.chi2.pdf(x,param[0], loc=param[1], scale=param[2])
plt.plot(x,pdf_fitted,label="Fit")
plt.legend()
plt.xlabel("Parallax uncertainties [mas]")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()







