# ****Checkich — v0.8****

## *Kaj je Checkich?*
Chechkich je namizna aplikacija za zaznavanje jamskih vhodov iz podatkov laserskega skeniranja (LiDAR). Iz oblakov točk v formatu LAZ/LAS samodejno poišče površinske depresije, ki ustrezajo jamskim vhodom, rezultate pa prikaže na interaktivnem zemljevidu in jih izvozi v standardne GIS formate.

## Funkcionalnosti
*_Zaznavanje_*
•	Nalaganje datotek LAZ in LAS
•	Izgradnja digitalnega modela reliefa (DMR) iz talnih točk
•	Analiza globine depresij glede na okoliški teren
•	Prostorsko grupiranje kandidatnih (nizkih) točk z algoritmom DBSCAN
•	Izbirna vključitev posamičnih (negrupiranih) nizkih točk

*Vizualizacija*
•	Interaktivni zemljevid z zaznanimi jami (Folium/Leaflet)
•	Prikaz znanih jam iz GEOJSON za primerjavo
•	3D prikaz oblaka točk in triangulacijske površine (TIN)
•	Barvno kodiranje po globini in zaupanju zaznave

*Analiza*
•	Tabela gruč z geometrijskimi metrikami (površina, obseg, kompaktnost, razmerje stranic)
•	Razvrstitev oblik vhodov: okrogla, ovalna, podolgovata, posamezna točka, par točk
•	Ocena zaupanja za vsako zaznanо gručo

*Izvoz*
•	Shapefile točk zaznanih vhodov
•	Shapefile gruč z vsemi metrikami
•	Interaktivna HTML karta
•	Statistike v formatu JSON

## **Navodila za uporabo**
### 0. Zagon
*Prva namestitev:* Prenesi vso vsebino repozitorija in jo odzipaj na željeno mesto. V naslovni vrstici v file explorerju izbriši pot do datoteke in napiši CMD in stisni enter, odpre se comand prompt v katerega napišeš: _python -m venv .venv_ Ko se ustvari virtualno okolje ga aktiviraš in vneseš _.venv\Scripts\activate_  in stisneš enter. Vse potrebne knjižnice namestiš tako da vneseš: _pip install -r requirements.txt_ in stisneš enter.  Nato napišeš še: _python checkich.py_ stisneš enter. Zagon traja nekaj 10 s, nato se pojavi uporabniški vmesnik.

*Kasnejši zagoni:* Pri kasnejših zagonih je potrebno samo aktivirati virtualno okolje in pognati skripto. V CMD vstaviš _.venv\Scripts\activate_ in nato z ukazom _python checkich.py_ poženeš aplikacijo
### 1. Nalaganje podatkov
Kliknite Brskaj in izberite datoteko LAZ ali LAS. Podprte so datoteke standardnega formata LAS 1.2–1.4 s klasificiranimi talnimi točkami (razred 2).
### 2. Izbira mape za izvoz
V polju Output določite mapo, kamor se bodo shranili rezultati (shapefileji, karte, statistike).
### 4. Nastavitev parametrov
|Parameter	|Opis	|Priporočena vrednost|
|-------------|-------------|-------------|
|Občutljivost	|Prednastavitev treh tipov detekcije (konzervativno /srednje/ občutljivo)|	Srednje za večino primerov|
|Radius (m)	|Polmer analiznega okna okoli vsake točke|	5 m|
|globina (m)	|Minimalna globina depresije za zaznavo|	3 m|
|vključi osamele točke|Vključi izolirane točke brez sosedov	|Izklopljeno|


### 5. Zagon zaznavanja
Kliknite Začni detekcijo. Med obdelavo se prikazuje napredek. Čas obdelave je odvisen od velikosti datoteke — tipična datoteka (150 MB) se obdela v 1–2 minutah, odvisno od velikosti računalnikovega pomnilnika.
### 6. Pregled rezultatov
*Zavihek Map View:* Interaktivni zemljevid prikazuje zaznane vhode (barvno po globini) in znane jame iz baze SLO JAME 2024 (modri markerji). Kliknite na marker za podrobnosti.
*Zavihek 3D View:* Tridimenzionalni prikaz oblaka točk z interpolirano površino. Pogled lahko zavrtite z miško, povečate s kolescem.
*Podokno Najdene gruče:* Tabela vseh zaznananih gruč z metrikami: koordinate centroida, število točk, globina, površina, oblika vhoda in ocena zaupanja.
### 7. Izvoz
Avtomatsko se ob pogonu detekcije shranijo: 
•	detected_points.shp — posamezne zaznane točke
•	detected_clusters.shp — gruče z atributi
•	map_interactive.html — karta, ki jo odprete v brskalniku
•	statistics.json — povzetek zaznave
•	DMR iz talnih točk
________________________________________
## **Sistemske zahteve**
OS	Windows 10 / 11 (64-bit)
RAM	vsaj 8 GB+, za velike datoteke 16 GB+
Disk	~1 GB za aplikacijo + prostor za izhodne datoteke
Python 3.14.3 nameščen in dodan v path
