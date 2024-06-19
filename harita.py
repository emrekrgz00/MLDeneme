import osmnx as ox
import matplotlib.pyplot as plt

admin = ox.geocode_to_gdf('Budapest')
admin_poly = admin.geometry.values[0]
admin_poly

G = ox.graph_from_polygon(admin_poly)
a,b = G.number_of_nodes(), G.number_of_edges()

nodes, edges = ox.graph_to_gdfs(G)

plt.style.use('dark_background')
f, ax = plt.subplots(1,1,figsize=(10,10))





# import matplotlib.pyplot as plt

# # admin_poly'yi doldurarak ekrana basma
# plt.figure(figsize=(10, 8))
# x, y = admin_poly.exterior.xy
# plt.fill(x, y, color='lightblue', alpha=0.6)  # Çokgeni lightblue rengiyle doldur, %60 şeffaflık
# plt.plot(x, y, color='blue', alpha=0.7, linewidth=0.7)  # Çokgenin kenarını mavi renkte çiz
# plt.title('Budapest Admin Boundary')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.grid(True)
# plt.show()

