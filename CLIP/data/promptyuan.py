
KL_PROMPTS = {
    "KneeOA": {
        "templates": [
            "knee X-ray showing {}"
        ],

        "slide_classnames": [
            # 每个类别一个列表，内含多条描述
            [
                "Bilateral knee joint spaces are symmetric with smooth articular surfaces; no osteophyte formation is observed. Trabecular bone is intact without subchondral sclerosis or bone degeneration, and overall skeletal alignment is normal.",
                "X-ray demonstrates uniform bone density in both knees, normal joint space width, smooth joint margins, absence of osteophytes, intact subchondral bone, and no degenerative changes or skeletal deformities.",
                "Bone density is homogeneous across all knee compartments, joint spaces are preserved, articular surfaces are smooth, no bony outgrowths are seen, subchondral bone shows no sclerosis or degeneration, and the mechanical axis is physiologically aligned.",
                "Imaging shows normal knee joint architecture, symmetrical intercondylar space, no osteophytes or marginal spurs, intact subchondral bone without sclerosis or degenerative changes, and regular bone morphology with continuous trabecular pattern.",
                "X-ray evaluation reveals uniform bone density, consistent joint space width, smooth joint margins, absence of osteophytes, intact subchondral bone, normal skeletal alignment, and regular bone morphology.",
                "Bilateral knee X-rays show clear, symmetric joint spaces, no osteophytes or marginal bony projections, intact subchondral bone, no bone destruction, and normal skeletal morphology.",
                "Radiographs demonstrate uniform and dense bone, normal joint space, smooth articular surfaces, no osteophytes or abnormal trabecular sclerosis, intact subchondral bone, and normal bone alignment.",
                "X-ray shows complete knee joint anatomy, symmetric joint space width, no osteophytes or marginal bony proliferation, subchondral bone without sclerosis, uniform bone density, and normal bone morphology.",
                "Knee radiographs show smooth and even joint spaces, no bony outgrowths or irregular proliferations, subchondral bone without sclerosis, clear trabecular pattern, proper bone alignment, and no degenerative signs.",
                "Imaging assessment shows uniform dense bone, preserved joint spaces, smooth articular surfaces, no osteophytes or marginal bony growth, intact subchondral bone, and normal bone morphology.",
                "X-ray demonstrates continuous dense bone in both knees, symmetric joint space, smooth and regular joint margins, absence of osteophytes or bone proliferation, intact subchondral layer, and physiologically normal skeletal alignment.",
                "Bilateral knee X-rays show clear and even joint spaces, smooth bone ends, no osteophyte formation, subchondral bone without sclerosis or degenerative changes, uniform bone density, and normal bone morphology and mechanical axis.",
            ],  # KL-0

            [
                "Mild marginal osteophyte formation is noted at the joint margins with subtle, questionable narrowing of the joint space; subchondral bone remains without sclerosis, and bone alignment is preserved.",
                "X-ray demonstrates minimal bony outgrowth along the tibiofemoral margins, with a possible early reduction in joint space; trabecular pattern is intact, and no gross deformity is observed.",
                "Slight lip-shaped osteophytes are evident at the periphery of the knee joint, accompanied by a subtle trend toward joint space narrowing; subchondral bone density is normal, and skeletal morphology is maintained.",
                "Imaging reveals minor marginal bony spurs at the femoral condyles, with joint space appearing mildly reduced but still largely preserved; no subchondral sclerosis or bone erosion is present.",
                "Radiographs show early peripheral osteophytic changes along the medial and lateral joint margins, with joint space demonstrating a tentative narrowing; bone trabeculae remain continuous, and overall bone alignment is normal.",
                "Small marginal osteophytes are observed at the tibial plateau edges, with joint space showing a subtle decrease suggestive of early degenerative change; subchondral bone remains clear without sclerosis.",
                "X-ray evaluation reveals faint lip-like bony projections along joint margins with a mildly reduced joint space, while subchondral bone and articular surfaces remain otherwise unremarkable.",
                "Minimal osteophyte formation is seen at the joint periphery with questionable joint space narrowing; bone density and trabecular architecture are preserved, and no deformity is evident.",
                "Imaging demonstrates subtle peripheral bony spurs with a slight tendency toward narrowing of the tibiofemoral joint space; subchondral bone exhibits no sclerosis, and mechanical axis is maintained.",
                "Early osteophytic changes are noted along the joint edges with joint space appearing marginally decreased; trabecular bone remains uniform, and there is no evidence of structural deformity.",
                "Faint lip-shaped osteophytes are present at the joint margins, accompanied by a mild, equivocal reduction in joint space width; subchondral bone and cortical outlines are intact.",
                "Radiographs show early peripheral bony projections with possible early narrowing of joint space; bone density is normal, articular surfaces are smooth, and skeletal alignment remains physiologic.",

            ],  # KL-1

            [
                "X-ray demonstrates definite marginal osteophyte formation along the femoral and tibial joint margins, with joint space largely preserved and subchondral bone density remaining normal.",
                "Clear peripheral bony spurs are evident at the knee joint, while joint space width appears essentially maintained; no subchondral sclerosis or bone deformity is observed.",
                "Radiographs reveal prominent marginal osteophytes with only minimal, if any, reduction of the tibiofemoral joint space; trabecular bone architecture is intact.",
                "Imaging shows distinct lip-shaped osteophytes along the joint edges, joint space is largely preserved, and subchondral bone demonstrates no sclerosis or degenerative change.",
                "Moderate peripheral bony projections are seen at both medial and lateral joint margins, with joint space maintained; bone contours and trabecular pattern remain normal.",
                "Knee X-rays display definite marginal osteophyte development with joint space essentially normal; subchondral bone and overall skeletal alignment are preserved.",
                "Distinct osteophytes are noted along the articular margins, accompanied by only subtle or absent narrowing of the joint space; no subchondral sclerosis or bone erosion is present.",
                "Imaging demonstrates clear bony outgrowths at the joint periphery, joint space width is maintained, and bone density and morphology remain unremarkable.",
                "Radiographs reveal well-defined marginal osteophytes, with joint space largely preserved and trabecular bone structure intact; no deformity is evident.",
                "Peripheral lip-shaped osteophytes are present along joint margins, joint space shows minimal to no narrowing, and subchondral bone remains normal in density and contour.",
                "Knee X-ray demonstrates prominent osteophytic formation along the tibial and femoral edges, with joint space preserved and cortical and trabecular bone unaffected.",
                "Definite marginal bony spurs are observed with joint space essentially maintained; subchondral bone and overall skeletal alignment are physiologically normal.",

            ],  # KL-2

            [
                "X-ray demonstrates multiple marginal osteophytes along the femoral and tibial condyles, with clear narrowing of the joint space and subchondral sclerosis; mild bone remodeling is present without significant deformity.",
                "Imaging reveals pronounced bony spurs at the joint margins, marked reduction in tibiofemoral joint space, subchondral bone thickening, and early trabecular changes; articular alignment remains largely preserved.",
                "Radiographs show multiple peripheral osteophytes with definite joint space narrowing, subchondral sclerosis, and mild bone density irregularities; no severe bone contour deformity is observed.",
                "Knee X-ray demonstrates distinct osteophytic formations along the articular margins, moderate joint space narrowing, subchondral bone hardening, and early signs of bone remodeling; bone morphology remains generally normal.",
                "Imaging reveals multifocal marginal osteophytes, narrowed joint space, subtle subchondral sclerosis, and mild trabecular irregularity; mechanical axis alignment is maintained.",
                "Radiographs show multiple lip-shaped osteophytes at the joint periphery, moderate reduction of joint space, subchondral bone densification, and mild bone degenerative changes; no significant malalignment is noted.",
                "X-ray demonstrates several peripheral bony spurs, clear narrowing of the tibiofemoral space, early subchondral sclerosis, and mild trabecular changes; articular surfaces are otherwise maintained.",
                "Imaging reveals multifocal osteophytes along the femoral and tibial margins, joint space narrowing, subchondral bone thickening, and mild bone density heterogeneity; no marked deformity of bone ends is evident.",
                "Radiographs show pronounced marginal osteophyte formation, significant narrowing of joint space, subchondral sclerosis, and early degenerative trabecular changes; skeletal alignment is largely preserved.",
                "Knee X-ray demonstrates multiple peripheral osteophytes, moderately narrowed joint space, subchondral hardening, and mild bone remodeling; articular surfaces remain largely intact.",
                "Imaging reveals multifocal osteophytic outgrowths along joint margins, clear joint space reduction, subchondral bone sclerosis, and subtle trabecular irregularities; no severe bone end deformity is seen.",
                "Radiographs demonstrate several lip-shaped osteophytes, definite joint space narrowing, subchondral sclerosis, and mild degenerative changes in bone structure; overall alignment and morphology remain largely physiologic.",

            ],  # KL-3

            [
                "X-ray demonstrates extensive, multiple marginal osteophytes with severe joint space narrowing approaching complete obliteration; pronounced subchondral sclerosis and marked bone remodeling are present, accompanied by significant deformity of the bone ends.",
                "Imaging reveals diffuse osteophytic proliferation along the femoral and tibial margins, almost complete loss of joint space, severe subchondral bone sclerosis, and advanced trabecular degeneration with bone end malalignment.",
                "Radiographs show widespread lip-shaped and irregular osteophytes, markedly reduced joint space, severe subchondral densification, and significant degenerative changes in bone structure, with deformity of the articular surfaces.",
                "Knee X-ray demonstrates extensive peripheral bony outgrowths, near-total joint space obliteration, pronounced subchondral sclerosis, severe bone degeneration, and distorted morphology of femoral and tibial condyles.",
                "Imaging reveals multiple large osteophytes along all joint margins, joint space almost absent, subchondral bone markedly sclerotic, trabecular bone shows advanced degeneration, and bone ends are deformed.",
                "Radiographs show diffuse marginal osteophytes, severe narrowing of tibiofemoral space, significant subchondral sclerosis, marked trabecular degeneration, and deformity of the bone contours.",
                "X-ray demonstrates extensive osteophytic formation, near-complete loss of joint space, dense subchondral sclerosis, advanced degenerative bone changes, and pronounced bone end malformation.",
                "Imaging reveals multiple irregular osteophytes, joint space nearly obliterated, severe subchondral densification, significant bone remodeling, and deformity of femoral and tibial articular surfaces.",
                "Radiographs show widespread peripheral bony spurs, markedly narrowed or absent joint space, profound subchondral sclerosis, advanced trabecular degeneration, and distorted skeletal morphology.",
                "Knee X-ray demonstrates extensive osteophytes along medial and lateral joint margins, joint space almost completely lost, severe subchondral hardening, pronounced bone density changes, and malformation of articular ends.",
                "Imaging reveals diffuse marginal and central osteophytes, near-total obliteration of joint space, severe subchondral sclerosis, advanced bone degeneration, and deformity of femoral and tibial condyles.",
                "Radiographs show multiple large osteophytes, joint space nearly absent, profound subchondral bone sclerosis, marked trabecular degeneration, and significant bone end deformation.",

            ],  # KL-4
        ]
    }
}
