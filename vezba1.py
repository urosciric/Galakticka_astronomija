from astroquery.gaia import Gaia
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import Angle

##############################################################

# zvezde spektralnog tipa O (Teff >= 30,000 K)
# uzimamo samo 2000

o_query = """
SELECT TOP 3000
    g.source_id, 
    g.ra, 
    g.dec, 
    g.parallax, 
    g.parallax_over_error, 
    g.pmra, 
    g.pmdec, 
    g.phot_g_mean_mag, 
    g.bp_rp,
    a.teff_gspphot
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.astrophysical_parameters AS a ON g.source_id = a.source_id
WHERE a.teff_gspphot >= 30000
  AND g.parallax > 0
  AND g.parallax_over_error > 5
"""

job_o = Gaia.launch_job_async(o_query)
df_o = job_o.get_results().to_pandas()
df_o.to_csv("o_stars.csv", index=False)

##############################################################

# učitavam podatke za O: ra, dec i parallax
df_o = pd.read_csv("o_stars.csv")
o_data = df_o[["ra", "dec", "parallax"]]

##############################################################

# učitavamo celi katalog - Vizier ima svoj row limit
Vizier.ROW_LIMIT = -1

# globularna jata
globular_result = Vizier.get_catalogs("VII/202")
df_globular = globular_result[0].to_pandas()

# zadržavam samo podatke koji su mi potrebni
df_globular_clean = df_globular[["ID", "RAJ2000", "DEJ2000"]]
df_globular_clean.to_csv("globular_clusters_harris.csv", index=False)

# otvorena jata
open_result = Vizier.get_catalogs("B/ocl")
df_open = open_result[0].to_pandas()

# zadržavam samo podatke koji su mi potrebni
df_open_clean = df_open[["RAJ2000", "DEJ2000", "Dist"]]
df_open_clean.to_csv("open_clusters_dias.csv", index=False)

############################################################## ZADATAK 1
# U prvom zadatku ćemo da napravimo scatter plot deklinacije od rektascenzije za otvorena jata i O zvezde.
# Funkcija Angle iz astropy.coordinates konvertuje rektascenziju iz sat/min/sek u stepene, a deklinaciju iz stepeni/min/sek u stepene.
# Koordinate O zvezda su već u stepene, pa ih ne treba konvertovati.
##############################################################
# konverzija koordinata otvorenih jata u stepene
# RA je u satima/min/sek, dec u stepenima/min/sek
ra_zvez  = Angle(df_open["RAJ2000"].astype(str).values, unit=u.hourangle).deg
dec_zvez = Angle(df_open["DEJ2000"].astype(str).values, unit=u.deg).deg

# Gaia koordinate su već u stepenima
ra_o  = df_o["ra"]
dec_o = df_o["dec"]

# plotujemo otvorena jata i O zvezde
'''
plt.scatter(ra_zvez, dec_zvez, label='Zvezdana jata', color='lime', marker='^')
plt.scatter(ra_o, dec_o, label='O zvezde', color='indianred', marker='o')

plt.title('Grafik deklinacije od rektascenzije')
plt.xlabel=('rektascenzija')
plt.ylabel=('deklinacija')
plt.legend()
plt.grid(True)
plt.show()
'''
############################################################## ZADATAK 2
# Gaia DR3 daje parallax u milisekundama (mas), a udaljenost u parsecima je 1/parallax (u sekundama).
# Dakle, da bismo dobili udaljenost u parsecima, moramo podeliti parallax sa 1000.
##############################################################
udaljenost = 1 / (df_o["parallax"] / 1000)  # parallax u masama, udaljenost u parsecima
print(udaljenost)

############################################################## ZADATAK 3
# Podaci za O zvezde i otvorena jata su u stepenima, pa možemo da ih spojimo u jedan niz za fitovanje velikog kruga.
# Veliki krug je definisan sa dva parametra: alfa_w (rektascenzija čvora) i nagib (nagib velikog kruga).
# Parametri se dobijaju fitovanjem funkcije velikog kruga na podatke rektascenzije i deklinacije.
##############################################################

from scipy.optimize import curve_fit

# Svi podaci zajedno (u stepenima): O zvezde + otvorena jata
ra_svi  = np.concatenate([ra_o.values,  ra_zvez])
dec_svi = np.concatenate([dec_o.values, dec_zvez])

# Funkcija velikog kruga: tan(dec) = sin(ra - alfa_w) * tan(nagib)
def veliki_krug(ra, alfa_w, nagib):
    ra_rad = np.radians(ra)
    alfa_w_rad = np.radians(alfa_w)
    nagib_rad = np.radians(nagib)
    dec_rad = np.arctan(np.sin(ra_rad - alfa_w_rad) * np.tan(nagib_rad))
    return np.degrees(dec_rad)

# p0 je početna procena [alfa_w, nagib] blizu očekivanog rešenja
popt, pcov = curve_fit(veliki_krug, ra_svi, dec_svi, p0=[280, 60])
alfa_w_fit, nagib_fit = popt
greske = np.sqrt(np.diag(pcov))

print("Parametri fita:")
print("alfa_w =", round(alfa_w_fit, 3), "+/-", round(greske[0], 3), "stepeni")
print("nagib  =", round(nagib_fit, 3), "+/-", round(greske[1], 3), "stepeni")

# Koordinate severnog galaktičkog pola iz fita
ra_png_fit = (alfa_w_fit - 90) % 360     # alfa_w - 6h
dec_png_fit = 90 - nagib_fit

# Usvojene vrednosti za J2000
nagib_usv  = 62.87
alfa_w_usv = (18 + 51/60 + 26/3600) * 15  
ra_png_usv = alfa_w_usv - 90
dec_png_usv = 90 - nagib_usv

print("")
print("Severni galaktički pol:")
print("Fitovano: RA =", round(ra_png_fit, 3), " Dec =", round(dec_png_fit, 3))
print("Usvojeno: RA =", round(ra_png_usv, 3), " Dec =", round(dec_png_usv, 3))
print("Razlika:  dRA =", round(ra_png_fit - ra_png_usv, 3), " dDec =", round(dec_png_fit - dec_png_usv, 3))

# Pravimo liniju velikog kruga za plotovanje
ra_linija = np.linspace(0, 360, 1000)
dec_linija = veliki_krug(ra_linija, alfa_w_fit, nagib_fit)

plt.figure()
plt.scatter(ra_zvez, dec_zvez, label='Zvezdana jata', color='lime', marker='^')
plt.scatter(ra_o, dec_o, label='O zvezde', color='indianred', marker='o', s=10)
plt.plot(ra_linija, dec_linija, color='black', linewidth=2, label='Fit velikog kruga')

plt.title('Veliki krug na nebeskoj sferi')
plt.xlabel("rektascenzija")
plt.ylabel('deklinacija')
plt.xlim(0, 360)
plt.ylim(-90, 90)
plt.legend()
plt.grid(True)
plt.show()

############################################################## ZADATAK 4
# U ovom zadatku ćemo da prikažemo raspodelu globularnih jata u galaktičkom koordinatnom sistemu.
# Ustanovljeno je iz resenja da su globularna jata koncentrisana oko galaktičkog centra, odnosno da je raspodela jata u galaktičkoj ravni i da su O zvezde raspoređene duž velikog kruga koji je naget u odnosu na galaktičku ravan.
##############################################################
from astropy.coordinates import SkyCoord

# Pravimo objekat sa koordinatama svih globularnih jata.
# RAJ2000 je u satima/min/sek, DEJ2000 u stepenima/min/sek.
koord_glob = SkyCoord(
    ra=df_globular["RAJ2000"].astype(str).values,
    dec=df_globular["DEJ2000"].astype(str).values,
    unit=(u.hourangle, u.deg),
    frame="icrs"
)

# Pretvaramo u galaktički koordinatni sistem.
# l (galaktička longituda) i b (galaktička latituda).
gal = koord_glob.galactic

# Vadimo l i b u stepenima, kao obične numpy nizove.
# l ide u opsegu 0-360, b u opsegu -90 do 90.
l_glob = gal.l.deg
b_glob = gal.b.deg

plt.scatter(l_glob, b_glob, label='Globularna jata', color='blue', marker='o')
plt.xlabel('Galaktička longituda')
plt.ylabel('Galaktička latituda')
plt.legend()
plt.grid(True)
plt.show()

############################################################## ZADATAK 5
# Nebeska sfera je sferna površina, pa je potrebno da podelimo galaktičku latitudu b u zone jednakih površina.
# np.histogram() funkcija nam omogućava da brojimo koliko jata se nalazi u svakoj zoni, a zatim da izračunamo centar raspodele b_max.
# Latitudu b_max možemo da izračunamo za različite brojeve zona N, npr. N=10 i N=20, i da uporedimo rezultate.
##############################################################
import numpy as np
import matplotlib.pyplot as plt

def b_maks(b, N):
    # granice zona jednakih povrsina
    h_gran = np.linspace(-1, 1, N+1)
    b_gran = np.degrees(np.arcsin(h_gran))

    # centri zona
    h_cent = 0.5*(h_gran[:-1] + h_gran[1:])
    b_cent = np.degrees(np.arcsin(h_cent))

    n_b, _ = np.histogram(b, bins=b_gran) # broj jata u svakoj zoni
    b_max = np.sum(n_b*b_cent)/np.sum(n_b)

    plt.figure(figsize=(8, 5))
    plt.bar(b_cent, n_b, width=np.diff(b_gran))
    plt.axvline(b_max, color='red', linestyle='--')
    plt.xlabel('Galakticka latituda b')
    plt.ylabel('Broj globularnih jata')
    plt.title(f'Histogram po galaktičkoj latitudi ({N} zona jednakih površina)')
    plt.legend()
    plt.show()
    return b_max

b_max_10 = b_maks(b_glob, 10)
b_max_20 = b_maks(b_glob, 20)

print('b_max (10 zona) = {:.2f} deg'.format(b_max_10))
print('b_max (20 zona) = {:.2f} deg'.format(b_max_20))
############################################################## ZADATAK 6
# Historam galaktičke longituda l u zone jednakih širina, brojimo koliko jata se nalazi u svakoj zoni, i izračunavamo centar raspodele l_max.
##############################################################
# l_glob je u opsegu 0-360, pa ga prebacujemo u -180 do 180
l_wrap = np.where(l_glob > 180, l_glob - 360, l_glob)

def l_maks(l, N):
    l_gran = np.linspace(-180, 180, N+1) # zone jednake širine po l
    l_cent = 0.5*(l_gran[:-1] + l_gran[1:])

    n_l, _ = np.histogram(l, bins=l_gran)
    l_max = np.sum(n_l*l_cent)/np.sum(n_l)       

    plt.figure(figsize=(8, 5))
    plt.bar(l_cent, n_l, width=np.diff(l_gran), edgecolor='black')
    plt.axvline(l_max, color='red', linestyle='--', label=f'$l_{{max}}$ = {l_max:.2f}°')
    plt.xlabel('Galaktička longituda l')
    plt.ylabel('Broj globularnih jata')
    plt.title(f'Histogram po galaktickoj longitudi ({N} zona)')
    plt.legend()
    plt.show()
    return l_max

l_max_10 = l_maks(l_wrap, 10)
l_max_20 = l_maks(l_wrap, 20)

print('l_max (10 zona) = {:.2f} deg'.format(l_max_10))
print('l_max (20 zona) = {:.2f} deg'.format(l_max_20))
############################################################## ZADATAK 7
# Odstupanje izračunatog centra raspodele globularnih jata od Sgr A* (koordinate galaktičkog centra) u galaktičkom koordinatnom sistemu je vece od 1 stepena, što je značajno.
# To znači da centar raspodele globularnih jata nije u galaktičkom centru, već je pomeren u odnosu na njega, sto je posledica merenja samo 150 jata.
##############################################################
# centar prividne raspodele (iz zadataka 5 i 6, podela na 10 zona)
centar = SkyCoord(l=l_max_10*u.deg, b=b_max_10*u.deg, frame='galactic').icrs

# očekivane vrednosti za l = 0, b = 0
ocekivano = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic').icrs

sgr = SkyCoord('17h45m40.04s', '-29d00m28.1s', frame='icrs')

print('Izračunato: l = {:.2f} deg, b = {:.2f} deg'.format(l_max_10, b_max_10))
print('RA = {:.3f} deg, DEC = {:.3f} deg'.format(centar.ra.degree, centar.dec.degree))
print(' ' + centar.to_string('hmsdms'))
print('Očekivano (l=0, b=0): RA = {:.3f} deg, DEC = {:.3f} deg'.format(ocekivano.ra.degree, ocekivano.dec.degree))
print('Sgr A*: RA = {:.3f} deg, DEC = {:.3f} deg'.format(sgr.ra.degree, sgr.dec.degree))
print('Ugaono odstupanje izracunatog centra od Sgr A*: {:.2f} deg'.format(centar.separation(sgr).degree))
