import gridstatus


caiso = gridstatus.CAISO()

print(caiso.get_fuel_mix("today"))