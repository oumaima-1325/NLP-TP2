import requests
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionCatFact(Action):
    def name(self) -> Text:
        return "action_cat_fact"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Fetch a random cat fact from the Cat Facts API
        response = requests.get("https://catfact.ninja/fact")
        if response.status_code == 200:
            fact = response.json()['fact']
            dispatcher.utter_message(text=fact)
        else:
            dispatcher.utter_message(text="Oops! Couldn't fetch a cat fact at the moment. Please try again later.")
        return []
