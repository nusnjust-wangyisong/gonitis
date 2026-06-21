try:
    from data.promptyuan import KL_PROMPTS
except Exception:
    KL_PROMPTS = {
        "KneeOA": {
            "templates": [
                "knee X-ray showing {}"
            ],
            "slide_classnames": [
                [
                    "grade 0 (none): definite absence of x-ray changes of osteoarthritis",
                ],
                [
                    "grade 1 (doubtful): doubtful joint space narrowing and possible osteophytic lipping",
                ],
                [
                    "grade 2 (minimal): definite osteophytes and possible joint space narrowing",
                ],
                [
                    "grade 3 (moderate): moderate multiple osteophytes, definite narrowing of joint space, some sclerosis and possible deformity of bone ends",
                ],
                [
                    "grade 4 (severe): large osteophytes, marked narrowing of joint space, severe sclerosis and definite deformity of bone ends",
                ],
            ]
        }
    }
