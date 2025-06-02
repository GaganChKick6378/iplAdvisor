import openai
import re
from datetime import datetime
from data_fetcher import CricketDataFetcher
from database import VectorDatabase
from langsmith import traceable
from langchain_openai import ChatOpenAI 
import config

class FantasyAdvisor:
    def __init__(self):
        self.data_fetcher = CricketDataFetcher()
        self.vector_db = VectorDatabase()
        if config.OPENAI_API_KEY:
             openai.api_key = config.OPENAI_API_KEY

    @traceable(name="update_knowledge_base", run_type="chain") 
    def update_knowledge_base(self):
        ipl_query = "Latest IPL cricket updates news player performance"
        ipl_resp = self.data_fetcher.get_player_data(ipl_query)
        ipl_results = ipl_resp["results"]
        for item in ipl_results:
            if hasattr(item, "text") and item.text:
                self.vector_db.add_document(
                    item.text,
                    {"title": item.title, "url": item.url, "date": datetime.now().isoformat()}
                )
        inj_resp = self.data_fetcher.get_injury_updates()
        inj_results = inj_resp["results"]
        for item in inj_results:
            if hasattr(item, "text") and item.text:
                self.vector_db.add_document(
                    item.text,
                    {"title": item.title, "url": item.url, "type": "injury", "date": datetime.now().isoformat()}
                )
        return {"status": "success", "docs_added": len(ipl_results) + len(inj_results)}

    @traceable(name="_assess_data_quality", run_type="chain") 
    def _assess_data_quality(self, context: str):
        length_score = min(len(context) / 10000, 1.0)
        stats_pattern = r'\b\d+(?:\.\d+)?\s*(?:runs|wickets|average|strike rate|economy|points)\b'
        stats_matches = re.findall(stats_pattern, context.lower())
        stats_score = min(len(stats_matches) / 20, 1.0)
        current_year = datetime.now().year
        year_pattern = r'\b(20\d{2})\b'
        years = [int(y) for y in re.findall(year_pattern, context)]
        recency_score = 0.5
        if years:
            recent_years = [y for y in years if y >= current_year - 1]
            recency_score = len(recent_years) / max(len(years), 1)
        data_quality = 0.6 * length_score + 0.4 * stats_score
        data_recency = recency_score
        data_relevance = stats_score
        return data_quality, data_recency, data_relevance

    @traceable(name="get_player_recommendation", run_type="chain") 
    def get_player_recommendation(self, player_name: str):
        player_resp = self.data_fetcher.get_player_data(player_name)
        results = player_resp["results"]
        context = ""
        sources = 0
        for item in results:
            if hasattr(item, "text") and item.text:
                context += f"Source: {item.title}\n{item.text}\n\n"
                sources += 1
        data_quality, data_recency, data_relevance = self._assess_data_quality(context)
        
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set in config.")
            
        llm = ChatOpenAI(model_name=config.MODEL_NAME, openai_api_key=config.OPENAI_API_KEY) 

        prompt_str = (
            f"You are a fantasy cricket expert advisor for IPL.\n"
            f"Based on the following information about {player_name}, should I include them in my fantasy team?\n"
            f"Include strengths, weaknesses, and specific statistics.\n\n{context}"
        )

        recommendation_obj = llm.invoke(prompt_str)
        recommendation = recommendation_obj.content 

        uncertainty_phrases = [
            "uncertain", "unclear", "might", "may", "could be",
            "possibly", "perhaps", "not enough data", "limited information"
        ]
        certainty_factor = 1.0
        for phrase in uncertainty_phrases:
            if phrase in recommendation.lower():
                certainty_factor *= 0.9
        confidence = (
            0.4 * data_quality + 0.3 * data_recency + 0.3 * data_relevance
        ) * certainty_factor
        confidence = max(0.0, min(confidence, 1.0))
        label = (
            "High Confidence" if confidence >= config.HIGH_CONFIDENCE else
            "Medium Confidence" if confidence >= config.MEDIUM_CONFIDENCE else
            "Low Confidence" if confidence >= config.LOW_CONFIDENCE else
            "Very Low Confidence"
        )
        return {
            "recommendation": recommendation,
            "confidence_score": confidence,
            "confidence_label": label,
            "sources_used": sources,
            "data_quality": data_quality,
            "data_recency": data_recency,
            "data_relevance": data_relevance
        }

    @traceable(name="get_team_advice", run_type="chain") 
    def get_team_advice(self, team_name: str):
        team_resp = self.data_fetcher.get_team_news(team_name)
        items = team_resp["results"]
        context = ""
        sources = 0
        for item in items:
            if hasattr(item, "text") and item.text:
                context += f"Source: {item.title}\n{item.text}\n\n"
                sources += 1
        data_quality, data_recency, data_relevance = self._assess_data_quality(context)

        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set in config.")

        llm = ChatOpenAI(model_name=config.MODEL_NAME, openai_api_key=config.OPENAI_API_KEY) 
        prompt_str = (
            f"You are a fantasy cricket expert advisor for IPL.\n"
            f"Based on the following information about {team_name}, which players from this team should I consider for my fantasy team?\n"
            f"List the top 3–5 players with reasons:\n\n{context}"
        )
        advice_obj = llm.invoke(prompt_str)
        advice = advice_obj.content

        uncertainty_phrases = [
            "uncertain", "unclear", "might", "may", "could be",
            "possibly", "perhaps", "not enough data", "limited information"
        ]
        certainty_factor = 1.0
        for phrase in uncertainty_phrases:
            if phrase in advice.lower():
                certainty_factor *= 0.9
        confidence = (
            0.4 * data_quality + 0.3 * data_recency + 0.3 * data_relevance
        ) * certainty_factor
        confidence = max(0.0, min(confidence, 1.0))
        label = (
            "High Confidence" if confidence >= config.HIGH_CONFIDENCE else
            "Medium Confidence" if confidence >= config.MEDIUM_CONFIDENCE else
            "Low Confidence" if confidence >= config.LOW_CONFIDENCE else
            "Very Low Confidence"
        )
        return {
            "advice": advice,
            "confidence_score": confidence,
            "confidence_label": label,
            "sources_used": sources,
            "data_quality": data_quality,
            "data_recency": data_recency,
            "data_relevance": data_relevance
        }

    @traceable(name="get_captain_recommendation", run_type="chain") 
    def get_captain_recommendation(self, player_options: list):
        context = ""
        total_sources = 0
        for player in player_options:
            player_resp = self.data_fetcher.get_player_data(player)
            for item in player_resp["results"][:2]:
                if hasattr(item, "text") and item.text:
                    snippet = item.text[:1000]
                    context += f"Player: {player}\nSource: {item.title}\n{snippet}\n\n"
                    total_sources += 1
        data_quality, data_recency, data_relevance = self._assess_data_quality(context)
        
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set in config.")

        llm = ChatOpenAI(model_name=config.MODEL_NAME, openai_api_key=config.OPENAI_API_KEY) 
        prompt_str = (
            f"You are a fantasy cricket expert advisor for IPL.\n"
            f"Given these players: {', '.join(player_options)}, who should I select as captain for my fantasy team?\n"
            f"Compare their recent form with statistics:\n\n{context}"
        )
        recommendation_obj = llm.invoke(prompt_str)
        recommendation = recommendation_obj.content 
        
        uncertainty_phrases = [
            "uncertain", "unclear", "might", "may", "could be",
            "possibly", "perhaps", "not enough data", "limited information"
        ]
        certainty_factor = 1.0
        for phrase in uncertainty_phrases:
            if phrase in recommendation.lower():
                certainty_factor *= 0.9
        confidence = (
            0.4 * data_quality + 0.3 * data_recency + 0.3 * data_relevance
        ) * certainty_factor
        confidence = max(0.0, min(confidence, 1.0))
        label = (
            "High Confidence" if confidence >= config.HIGH_CONFIDENCE else
            "Medium Confidence" if confidence >= config.MEDIUM_CONFIDENCE else
            "Low Confidence" if confidence >= config.LOW_CONFIDENCE else
            "Very Low Confidence"
        )
        return {
            "recommendation": recommendation,
            "confidence_score": confidence,
            "confidence_label": label,
            "players_analyzed": player_options,
            "sources_used": total_sources,
            "data_quality": data_quality,
            "data_recency": data_recency,
            "data_relevance": data_relevance
        }

    @traceable(name="get_match_analysis", run_type="chain")
    def get_match_analysis(self, team1: str, team2: str):
        match_resp = self.data_fetcher.get_match_predictions(team1, team2)
        results = match_resp["results"]
        context = ""
        sources = 0
        for item in results:
            if hasattr(item, "text") and item.text:
                context += f"Source: {item.title}\n{item.text}\n\n"
                sources += 1
        data_quality, data_recency, data_relevance = self._assess_data_quality(context)

        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set in config.")
            
        llm = ChatOpenAI(model_name=config.MODEL_NAME, openai_api_key=config.OPENAI_API_KEY)
        prompt_str = (
            f"You are a fantasy cricket expert advisor for IPL.\n"
            f"Analyze the upcoming match between {team1} and {team2}. Provide fantasy recommendations including key players from both teams.\n\n{context}"
        )
        analysis_obj = llm.invoke(prompt_str)
        analysis = analysis_obj.content 

        uncertainty_phrases = [
            "uncertain", "unclear", "might", "may", "could be",
            "possibly", "perhaps", "not enough data", "limited information"
        ]
        certainty_factor = 1.0
        for phrase in uncertainty_phrases:
            if phrase in analysis.lower():
                certainty_factor *= 0.9
        confidence = (
            0.4 * data_quality + 0.3 * data_recency + 0.3 * data_relevance
        ) * certainty_factor
        confidence = max(0.0, min(confidence, 1.0))
        label = (
            "High Confidence" if confidence >= config.HIGH_CONFIDENCE else
            "Medium Confidence" if confidence >= config.MEDIUM_CONFIDENCE else
            "Low Confidence" if confidence >= config.LOW_CONFIDENCE else
            "Very Low Confidence"
        )
        return {
            "analysis": analysis,
            "confidence_score": confidence,
            "confidence_label": label,
            "teams_analyzed": [team1, team2],
            "sources_used": sources,
            "data_quality": data_quality,
            "data_recency": data_recency,
            "data_relevance": data_relevance
        }