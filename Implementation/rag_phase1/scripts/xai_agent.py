def justify_recommendation(plan, profile):
    return (
        f"We recommend a {plan['recommendation_type']} for {plan['next_concept']} "
        f"because your current knowledge score is {profile['knowledge'][plan['next_concept']]:.1f} "
        f"and you prefer {profile['preferences'][0]}."
    )
