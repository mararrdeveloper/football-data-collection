def get_full_name(name):
    #missmapics of the 2 databases
    name_dict = {
        "Man United" : "Manchester United",
        "Man City" : "Manchester City",
        "Wolves" : "Wolverhampton Wanderers",
        "Newcastle" : "Newcastle United",
        "West Brom" : "West Bromwich Albion",
        "QPR" : "Queens Park Rangers",
    }
    if name in name_dict:
        return name_dict[name]
    return name