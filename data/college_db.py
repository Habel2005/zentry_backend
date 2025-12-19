
class CollegeDB:
    def get_seat_info(self, program):
        if program == "B.Tech":
            return {"remaining": 18}
        return None

    def get_placement_info(self):
        return {"message": "Placements are consistently good with reputed companies."}
