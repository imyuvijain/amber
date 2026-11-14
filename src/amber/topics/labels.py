"""A fixed, generic vocabulary for naming topic clusters: ``(label, utility)``.

Utility labels are chores (checking the weather, finding a route), not interests.
Nothing here is about one person; the list is the same for everyone.
"""

LABELS: list[tuple[str, bool]] = [
    # games and play
    ("video games", False), ("esports", False), ("board games", False), ("chess", False),
    ("puzzles", False), ("game development", False),
    # music and sound
    ("music", False), ("music production", False), ("hip hop", False), ("rock music", False),
    ("classical music", False), ("electronic music", False), ("pop music", False),
    ("indian music", False), ("k-pop", False), ("guitar", False), ("piano", False),
    ("singing", False), ("music theory", False), ("concerts", False), ("podcasts", False),
    # screen
    ("anime", False), ("manga", False), ("movies", False), ("tv series", False),
    ("cartoons and animation", False), ("stand-up comedy", False), ("memes", False),
    ("celebrity news", False), ("film making", False), ("bollywood", False),
    # science and learning
    ("physics", False), ("astronomy and space", False), ("chemistry", False),
    ("biology", False), ("mathematics", False), ("statistics", False),
    ("history", False), ("geography", False), ("philosophy", False), ("psychology", False),
    ("economics", False), ("linguistics and languages", False), ("literature and books", False),
    ("poetry", False), ("medicine and health", False), ("neuroscience", False),
    ("environment and climate", False), ("engineering", False), ("architecture", False),
    ("exam preparation", False), ("university admissions", False), ("study tips", False),
    ("online courses", False),
    # tech
    ("programming", False), ("machine learning and ai", False), ("web development", False),
    ("computers and hardware", False), ("smartphones and gadgets", False),
    ("cybersecurity", False), ("data science", False), ("robotics", False),
    ("software tools", False), ("startups", False), ("cryptocurrency", False),
    # money and work
    ("personal finance", False), ("stock market investing", False), ("business", False),
    ("careers and jobs", False), ("entrepreneurship", False), ("marketing", False),
    # body and life
    ("fitness and gym", False), ("running", False), ("yoga and meditation", False),
    ("nutrition and diet", False), ("cooking", False), ("baking", False),
    ("food and restaurants", False), ("coffee", False), ("fashion", False),
    ("sneakers", False), ("skincare and grooming", False), ("hairstyles", False),
    ("relationships and dating", False), ("parenting", False), ("self improvement", False),
    ("productivity", False), ("mental health", False), ("religion and spirituality", False),
    # sport
    ("cricket", False), ("football", False), ("basketball", False), ("tennis", False),
    ("formula 1", False), ("martial arts", False), ("wrestling", False), ("olympics", False),
    ("cycling", False), ("swimming", False), ("badminton", False),
    # making
    ("drawing and painting", False), ("photography", False), ("graphic design", False),
    ("3d modelling", False), ("diy and crafts", False), ("woodworking", False),
    ("writing", False), ("video editing", False),
    # world
    ("travel", False), ("cars", False), ("motorcycles", False), ("aviation", False),
    ("trains", False), ("pets", False), ("dogs", False), ("cats", False), ("wildlife and nature", False),
    ("gardening", False), ("interior design", False), ("real estate", False),
    ("politics", False), ("world news", False), ("law", False), ("military", False),
    ("japan", False), ("korea", False), ("india", False), ("canada", False),
    ("social media", False), ("youtubers and vlogs", False), ("pranks and challenges", False),
    ("magic tricks", False), ("true crime", False), ("conspiracy theories", False),
    # chores
    ("weather", True), ("maps and directions", True), ("translation", True),
    ("time and time zones", True), ("unit conversion", True), ("calculator", True),
    ("email and accounts", True), ("passwords and login", True), ("online shopping", True),
    ("banking", True), ("phone and internet plans", True), ("public transport schedules", True),
    ("food delivery", True), ("tech support", True), ("downloads", True),
    ("dictionary definitions", True), ("spelling", True), ("school portal", True),
]
