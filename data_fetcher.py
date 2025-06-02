from exa_py import Exa
import config
from datetime import datetime
from langsmith import traceable

class CricketDataFetcher:
    def __init__(self):
        self.client = Exa(config.EXA_API_KEY)

    @traceable(name="get_player_data", run_type="tool")
    def get_player_data(self, player_name: str):
        results = self.client.search_and_contents(
            f"IPL cricket player {player_name} recent performance statistics",
            text={"max_characters": config.MAX_CHARS},
            num_results=config.MAX_RESULTS,
            type="auto"
        )
        return {"results": results.results}

    @traceable(name="get_team_news", run_type="tool") 
    def get_team_news(self, team_name: str):
        results = self.client.search_and_contents(
            f"IPL cricket team {team_name} recent news updates squad changes",
            text={"max_characters": config.MAX_CHARS},
            num_results=config.MAX_RESULTS,
            type="auto"
        )
        return {"results": results.results}

    @traceable(name="get_match_predictions", run_type="tool")
    def get_match_predictions(self, team1: str, team2: str):
        results = self.client.search_and_contents(
            f"IPL cricket match prediction {team1} vs {team2} analysis",
            text={"max_characters": config.MAX_CHARS},
            num_results=config.MAX_RESULTS,
            type="auto"
        )
        return {"results": results.results}

    @traceable(name="get_injury_updates", run_type="tool")
    def get_injury_updates(self):
        results = self.client.search_and_contents(
            "IPL cricket recent player injuries updates team changes",
            text={"max_characters": config.MAX_CHARS},
            num_results=config.MAX_RESULTS,
            type="auto"
        )
        return {"results": results.results}