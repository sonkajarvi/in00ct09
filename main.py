import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import folium
from streamlit_folium import st_folium

df_a = pd.read_csv("https://raw.githubusercontent.com/sonkajarvi/in00ct09/master/2025-02-26_linear-accelerometer.csv")
df_l = pd.read_csv("https://raw.githubusercontent.com/sonkajarvi/in00ct09/master/2025-02-26_location.csv")

st.title("Fysiikan loppuprojekti")

data = df_a["Z (m/s^2)"]
T = data.max()
N = len(data)
cutoff = 1 / 0.06

a, b = butter(3, cutoff / ((N / T) / 2), btype="low", analog=False)
filter = filtfilt(a, b, data)

steps = 0
for i in range(N - 1):
    steps += filter[i] / filter[i + 1] < 0

st.write("Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta", round(steps / 2))

dt = T / df_a["Time (s)"]
cutoff = 1 / 10

f = np.fft.fft(data, N)
freq = np.fft.fftfreq(N, dt)
f[np.abs(freq) > cutoff] = 0
i = np.fft.ifft(f)
fourier = i.real

t = 0
for i in range(N - 1):
    t += fourier[i] / fourier[i + 1] < 0

st.write("Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella:", round(t / 2))
st.write("Keskinopeus (GPS-datasta):", round(df_l["Velocity (m/s)"].mean(), 2), "m/s")

dist = (df_l["Velocity (m/s)"] * df_l["Time (s)"].diff()).sum()

st.write("Kuljettu matka (GPS-datasta):", round(dist, 2), "m")
st.write("Askelpituus (lasketun askelmäärän ja matkan perusteella)", round(dist / steps, 2), "m")

st.title("Suodatettu kiihtyvyysdata")

df_a["filter"] = filter
st.line_chart(df_a[10:], x="Time (s)", y="filter", y_label="Kiihtyvyys (m/s^2)", x_label="Aika (s)", )

st.title("Tehospektri")

psd = f * np.conj(f) / N
L = np.arange(1, int(N / 2))

df = pd.DataFrame(np.transpose(np.array([freq[L],psd[L].real])), columns=["freq", "psd"])
st.line_chart(df[:300], x="freq", y="psd" , y_label="Teho", x_label="Taajuus (Hz)")

st.title("Karttakuva")

lat = df_l["Latitude (°)"].mean()
lon = df_l["Longitude (°)"].mean() + 0.005
map = folium.Map(location = [lat, lon], zoom_start = 16)

folium.PolyLine(df_l[["Latitude (°)","Longitude (°)"]], color="red", weight=5).add_to(map)
st_map = st_folium(map, width=1200, height=600)
