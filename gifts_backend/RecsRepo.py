from preference_matcher.compare import get_recommendations


class RecsRepo:
    def get_recs(self, input_preferences):
        # Assuming the get_recommendations function is defined elsewhere and imported
        recommendations = get_recommendations(input_preferences)
        return recommendations
