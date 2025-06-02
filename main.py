from advisor import FantasyAdvisor
import argparse
import sys
import config
import os
from dotenv import load_dotenv
load_dotenv()



print("---- Environment Variables after loading .env ----")
print(f"LANGSMITH_API_KEY Loaded: {os.getenv('LANGSMITH_API_KEY') is not None}")
print(f"LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
print(f"OPENAI_API_KEY Loaded: {os.getenv('OPENAI_API_KEY') is not None}")
print("-------------------------------------------------")

def main():
    print("Starting Fantasy IPL Cricket Advisor...")

    parser = argparse.ArgumentParser(description='Fantasy IPL Cricket Advisor')
    parser.add_argument('--update', action='store_true', help='Update knowledge base')
    parser.add_argument('--player', type=str, help='Get recommendation for a specific player')
    parser.add_argument('--team', type=str, help='Get advice for picking players from a team')
    parser.add_argument('--captain', type=str, nargs='+', help='Get captain recommendation from list of players')
    parser.add_argument('--match', nargs=2, metavar=('TEAM1', 'TEAM2'), help='Get match analysis for TEAM1 vs TEAM2')

    print("Parsing arguments...")
    args = parser.parse_args()
    print(f"Arguments received: {args}")

    print("Initializing advisor...")
    try:
        advisor = FantasyAdvisor() 
        print("Advisor initialized successfully")
    except Exception as e:
        print(f"Error initializing advisor: {e}")
        sys.exit(1)

    if args.update:
        print("Updating knowledge base...")
        try:
            result = advisor.update_knowledge_base()
            print("Knowledge base updated successfully!")
            print(result)
        except Exception as e:
            print(f"Error updating knowledge base: {e}")

    if args.player:
        print(f"Getting recommendation for {args.player}...")
        try:
            rec = advisor.get_player_recommendation(args.player)
            print("\nRECOMMENDATION:")
            print(rec["recommendation"])
            print(f"Confidence: {rec['confidence_label']} ({rec['confidence_score']:.2f})")
        except Exception as e:
            print(f"Error getting player recommendation: {e}")

    if args.team:
        print(f"Getting advice for team {args.team}...")
        try:
            adv = advisor.get_team_advice(args.team)
            print("\nTEAM ADVICE:")
            print(adv["advice"])
            print(f"Confidence: {adv['confidence_label']} ({adv['confidence_score']:.2f})")
        except Exception as e:
            print(f"Error getting team advice: {e}")

    if args.captain:
        print(f"Getting captain recommendation from {', '.join(args.captain)}...")
        try:
            cap = advisor.get_captain_recommendation(args.captain)
            print("\nCAPTAIN RECOMMENDATION:")
            print(cap["recommendation"])
            print(f"Confidence: {cap['confidence_label']} ({cap['confidence_score']:.2f})")
        except Exception as e:
            print(f"Error getting captain recommendation: {e}")

    if args.match:
        team1, team2 = args.match
        print(f"Getting match analysis for {team1} vs {team2}...")
        try:
            analysis = advisor.get_match_analysis(team1, team2)
            print("\nMATCH ANALYSIS:")
            print(analysis["analysis"])
            print(f"Confidence: {analysis['confidence_label']} ({analysis['confidence_score']:.2f})")
        except Exception as e:
            print(f"Error getting match analysis: {e}")

    if not any([args.update, args.player, args.team, args.captain, args.match]):
        print("No action specified. Use --help to see available options.")

if __name__ == "__main__":
    main()