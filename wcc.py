QUESTIONS = {
    "Pressure": ["Staff face unrealistic targets or financial stress.", "Pay/benefits are seen as unfair."],
    "Opportunity": ["One person controls approval, payment and records.", "Audits are rare or predictable.",
                    "There is no safe anonymous reporting channel."],
    "Rationalization": ["\"Everyone does it\" is commonly heard.", "Past misconduct went unpunished.",
                        "Leaders model bending rules."],
}
CONTEXTS = ["Bribery / corruption", "Scholarship / fund manipulation", "Abuse of authority", "Embezzlement / fraud"]

def level(total):
    return "Low" if total < 33 else "Moderate" if total < 66 else "High"

def rehab_plan(context, lvl):
    base = ["Restitution / corrective action where applicable", "Ethics & integrity workshop", "Clear written code of conduct"]
    extra = {"Low": ["Annual refresher training"],
             "Moderate": ["CBT-style sessions on rationalizations", "Dual approval for payments", "Quarterly audit"],
             "High": ["Structured counselling / restorative justice", "Remove sole control over funds",
                      "Monthly independent audit", "Supervised reintegration with mentor", "Anonymous whistleblower line"]}
    ctx = {"Bribery / corruption": ["Job rotation", "E-procurement & transparency portal"],
           "Scholarship / fund manipulation": ["Independent beneficiary verification", "Public merit lists"],
           "Abuse of authority": ["Oversight committee", "Clear grievance path for subordinates"],
           "Embezzlement / fraud": ["Segregation of duties", "Surprise cash/stock counts"]}
    return base + extra[lvl] + ctx[context]
