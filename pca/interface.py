import tkinter as tk
from LogisticRegression_PCA import predict_severity

def calculate_severity():
    distance = float(distance_entry.get())
    temperature = float(temperature_entry.get())
    wind_chill = float(wind_chill_entry.get())
    humidity = float(humidity_entry.get())
    pressure = float(pressure_entry.get())
    visibility = float(visibility_entry.get())
    wind_speed = float(wind_speed_entry.get())
    precipitation = float(precipitation_entry.get())
    amenity = int(amenity_var.get())
    bump = int(bump_var.get())
    crossing = int(crossing_var.get())
    give_way = int(give_way_var.get())
    junction = int(junction_var.get())
    no_exit = int(no_exit_var.get())
    railway = int(railway_var.get())
    roundabout = int(roundabout_var.get())
    station = int(station_var.get())
    stop = int(stop_var.get())
    traffic_calming = int(traffic_calming_var.get())
    traffic_signal = int(traffic_signal_var.get())
    sunrise_sunset = 1 if sunrise_sunset_var.get() == "Day" else 0
    
    input_values = [distance, temperature, wind_chill, humidity, pressure, visibility, wind_speed, precipitation,
                    amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop,
                    traffic_calming, traffic_signal, sunrise_sunset]
    
    severity = predict_severity(input_values)
    
    severity_label.config(text="Severity: " + str(severity))

# Create the main window
window = tk.Tk()
window.title("Severity Calculator")

# Create input labels and entry fields
distance_label = tk.Label(window, text="Distance (M):")
distance_label.grid(row=0, column=0)
distance_entry = tk.Entry(window)
distance_entry.grid(row=0, column=1)

temperature_label = tk.Label(window, text="Temperature (C):")
temperature_label.grid(row=1, column=0)
temperature_entry = tk.Entry(window)
temperature_entry.grid(row=1, column=1)

wind_chill_label = tk.Label(window, text="Wind Chill (F):")
wind_chill_label.grid(row=2, column=0)
wind_chill_entry = tk.Entry(window)
wind_chill_entry.grid(row=2, column=1)

humidity_label = tk.Label(window, text="Humidity (%):")
humidity_label.grid(row=3, column=0)
humidity_entry = tk.Entry(window)
humidity_entry.grid(row=3, column=1)

pressure_label = tk.Label(window, text="Pressure (in):")
pressure_label.grid(row=4, column=0)
pressure_entry = tk.Entry(window)
pressure_entry.grid(row=4, column=1)

visibility_label = tk.Label(window, text="Visibility (mi):")
visibility_label.grid(row=5, column=0)
visibility_entry = tk.Entry(window)
visibility_entry.grid(row=5, column=1)

wind_speed_label = tk.Label(window, text="Wind Speed (mph):")
wind_speed_label.grid(row=6, column=0)
wind_speed_entry = tk.Entry(window)
wind_speed_entry.grid(row=6, column=1)

precipitation_label = tk.Label(window, text="Precipitation (in):")
precipitation_label.grid(row=7, column=0)
precipitation_entry = tk.Entry(window)
precipitation_entry.grid(row=7, column=1)

amenity_var = tk.IntVar()
amenity_checkbox = tk.Checkbutton(window, text="Amenity", variable=amenity_var)
amenity_checkbox.grid(row=8, column=0)

bump_var = tk.IntVar()
bump_checkbox = tk.Checkbutton(window, text="Bump", variable=bump_var)
bump_checkbox.grid(row=9, column=0)

crossing_var = tk.IntVar()
crossing_checkbox = tk.Checkbutton(window, text="Crossing", variable=crossing_var)
crossing_checkbox.grid(row=10, column=0)

give_way_var = tk.IntVar()
give_way_checkbox = tk.Checkbutton(window, text="Give Way", variable=give_way_var)
give_way_checkbox.grid(row=11, column=0)

junction_var = tk.IntVar()
junction_checkbox = tk.Checkbutton(window, text="Junction", variable=junction_var)
junction_checkbox.grid(row=12, column=0)

no_exit_var = tk.IntVar()
no_exit_checkbox = tk.Checkbutton(window, text="No Exit", variable=no_exit_var)
no_exit_checkbox.grid(row=13, column=0)

railway_var = tk.IntVar()
railway_checkbox = tk.Checkbutton(window, text="Railway", variable=railway_var)
railway_checkbox.grid(row=14, column=0)

roundabout_var = tk.IntVar()
roundabout_checkbox = tk.Checkbutton(window, text="Roundabout", variable=roundabout_var)
roundabout_checkbox.grid(row=15, column=0)

station_var = tk.IntVar()
station_checkbox = tk.Checkbutton(window, text="Station", variable=station_var)
station_checkbox.grid(row=16, column=0)

stop_var = tk.IntVar()
stop_checkbox = tk.Checkbutton(window, text="Stop", variable=stop_var)
stop_checkbox.grid(row=17, column=0)

traffic_calming_var = tk.IntVar()
traffic_calming_checkbox = tk.Checkbutton(window, text="Traffic Calming", variable=traffic_calming_var)
traffic_calming_checkbox.grid(row=18, column=0)

traffic_signal_var = tk.IntVar()
traffic_signal_checkbox = tk.Checkbutton(window, text="Traffic Signal", variable=traffic_signal_var)
traffic_signal_checkbox.grid(row=19, column=0)

sunrise_sunset_label = tk.Label(window, text="Sunrise/Sunset:")
sunrise_sunset_label.grid(row=20, column=0)
sunrise_sunset_var = tk.StringVar()
sunrise_sunset_dropdown = tk.OptionMenu(window, sunrise_sunset_var, "Day", "Night")
sunrise_sunset_dropdown.grid(row=20, column=1)

calculate_button = tk.Button(window, text="Calculate Severity", command=calculate_severity)
calculate_button.grid(row=21, column=0, columnspan=2)

severity_label = tk.Label(window, text="Severity: ")
severity_label.grid(row=22, column=0, columnspan=2)

# Start the main loop
window.mainloop()

