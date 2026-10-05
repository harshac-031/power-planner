import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

st.title("Power Planner: Where Should the Next Plant Go?")
st.write("These are REAL power plants from around the world. Let the computer sort them into "
         "regions, then decide where the next plant should be built!")

# Step 1: load the real power plant data (34,936 plants, World Resources Institute)
plants = pd.read_csv("power_plants.csv")
counts = plants.groupby(["country_long", "primary_fuel"]).size().reset_index(name="n")
counts = counts[counts["n"] >= 8]                    # we need enough plants to make groups

# Step 2: the student chooses a country, an energy type and how many regions
countries = sorted(counts["country_long"].unique())
country = st.selectbox("Country (try YOUR country!)", countries, index=countries.index("India"))
fuels = sorted(counts[counts["country_long"] == country]["primary_fuel"])
fuel = st.selectbox("Energy type", fuels, index=fuels.index("Solar") if "Solar" in fuels else 0)
k = st.slider("How many regions?", 2, 6, 3)

# Step 3: the computer groups nearby plants together (clustering!)
plants = plants[(plants["country_long"] == country) & (plants["primary_fuel"] == fuel)].copy()
plants["region"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(plants[["latitude", "longitude"]])

# Step 4: how much power does each region make?
regions = plants.groupby("region").agg(plants=("name", "count"), total_MW=("capacity_mw", "sum")).round()
weakest = regions["total_MW"].idxmin()
centre = plants[plants["region"] == weakest][["latitude", "longitude"]].mean()
share = regions["total_MW"].max() / regions["total_MW"].sum() * 100

# Step 5: satellite map. Dot size = power, dot colour = region, star = build here!
colours = ["red", "blue", "green", "orange", "violet", "gray"]
m = folium.Map(tiles="Esri.WorldImagery")
for _, p in plants.iterrows():
    folium.CircleMarker([p["latitude"], p["longitude"]], radius=min(max(4, p["capacity_mw"] ** 0.5 / 4), 25),
                        color=colours[p["region"]], fill=True, fill_opacity=0.7,
                        tooltip=f"{p['name']}: {p['capacity_mw']:,.0f} MW").add_to(m)
folium.Marker([centre["latitude"], centre["longitude"]], tooltip="Build here!",
              icon=folium.Icon(color="black", icon="star")).add_to(m)
m.fit_bounds([[plants["latitude"].min(), plants["longitude"].min()],
              [plants["latitude"].max(), plants["longitude"].max()]])
st_folium(m, height=500, returned_objects=[])
st.markdown("  ".join(f":{colours[i]}[●] Region {i + 1}" for i in range(k)))

regions.index = [f"Region {i + 1}" for i in regions.index]
st.dataframe(regions)
st.success(f"Build the next {fuel} plant near {centre['latitude']:.2f}, {centre['longitude']:.2f}: "
           f"{regions.index[weakest]} has the LEAST {fuel} power.")
st.metric("Power made by the biggest region", f"{share:.0f}%")
