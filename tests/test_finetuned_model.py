"""
Advanced test suite for Qwen3 fine-tuned model.
Tests problem validation flows, reformulation chains, and cross-module intelligence.

Run with: python test_finetuned_model.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.chat import ChatRequest, ModuleContext, ChatAction
from app.agents.conversational import ConversationalAgent
from app.registry import get_conversational_agent, get_retriever


# ============================================================
# Helpers
# ============================================================

def make_request(module_key: str, message: str, locale: str = "fr", filled_fields: list = None, empty_fields: list = None):
    return ChatRequest(
        module=ModuleContext(
            module_key=module_key,
            label=module_key,
            filled_fields=filled_fields or [],
            empty_fields=empty_fields or [],
        ),
        message=message,
        locale=locale,
        conversation_history=[],
    )


def print_result(test_name: str, passed: bool, reply_preview: str = "", details: str = ""):
    status = "PASS" if passed else "FAIL"
    marker = "+" if passed else "x"
    print(f"[{marker}] {status}: {test_name}")
    if reply_preview:
        print(f"    Reply: {reply_preview[:120]}...")
    if details and not passed:
        print(f"    Detail: {details}")
    return passed


# ============================================================
# TEST GROUP 1: Vague Problem → Challenge → Reformulate
# ============================================================

def test_vague_problem_challenge_reformulate():
    """User gives vague problem, AI challenges, user asks reformulation, AI gives specific version."""
    agent = get_conversational_agent()

    # Step 1: User gives vague problem about social media in Senegal
    req = make_request(
        "problem-statement",
        "Au Senegal on a un probleme avec les reseaux sociaux"
    )
    resp = agent.run(req)

    passed = True
    details = ""

    # AI should flag it as vague
    if "vague" not in resp.reply.lower() and "flou" not in resp.reply.lower() and "precis" not in resp.reply.lower():
        passed = False
        details = "AI did not flag the problem as vague"
    if "qui" not in resp.reply.lower():
        passed = False
        details += " | AI did not ask WHO is affected"

    print_result(
        "Vague problem should be challenged",
        passed,
        resp.reply,
        details,
    )

    # Step 2: User asks to reformulate focusing on students
    req2 = make_request(
        "problem-statement",
        "Reformule ma problematique, acces sur les eleves"
    )
    resp2 = agent.run(req2)

    passed2 = True
    details2 = ""

    # AI should give a specific reformulation with context
    reply_lower2 = resp2.reply.lower()
    if "eleve" not in reply_lower2 and "etudiant" not in reply_lower2:
        passed2 = False
        details2 = "Reformulation does not mention students"
    if "dakar" not in reply_lower2 and "senegal" not in reply_lower2:
        passed2 = False
        details2 += " | Reformulation does not mention Senegal/Dakar context"
    if "reseau" not in reply_lower2 and "social" not in reply_lower2 and "ecran" not in reply_lower2:
        passed2 = False
        details2 += " | Reformulation does not mention social media/screens"

    # Should propose concrete values
    has_apply = any(a.type == "apply_fields" for a in resp2.actions)

    print_result(
        "Reformulate with student focus gives specific response",
        passed2 and has_apply,
        resp2.reply,
        details2,
    )


# ============================================================
# TEST GROUP 2: Problem Validation - AI Knows What to Ask
# ============================================================

def test_problem_validation_knows_questions():
    """On validation page, AI should know what questions to ask the user."""
    agent = get_conversational_agent()

    # Scenario: User is on problem-validation page with a problem statement
    req = make_request(
        "problem-validation",
        "Mon probleme: Les eleves a Dakar perdent 2h par jour sur les reseaux sociaux au lieu d'etudier",
        filled_fields=[
            {"field_name": "problemStatement", "label": "Enonce", "is_filled": True,
             "content": "Les eleves a Dakar perdent 2h par jour sur les reseaux sociaux"},
        ],
        empty_fields=["evidence", "interviewsCount", "willingnessToPay"],
    )
    resp = agent.run(req)

    passed = True
    details = ""
    reply_lower = resp.reply.lower()

    # AI should ask about evidence/interviews
    if "interview" not in reply_lower and "temoin" not in reply_lower and "preuve" not in reply_lower and "evidence" not in reply_lower:
        passed = False
        details = "AI did not ask for evidence or interviews"

    # AI should ask about frequency validation
    if "frequence" not in reply_lower and "combien" not in reply_lower and "fois" not in reply_lower and "jour" not in reply_lower:
        passed = False
        details += " | AI did not ask to validate frequency"

    # AI should ask about willingness to pay or urgency
    if "paier" not in reply_lower and "urgent" not in reply_lower and "important" not in reply_lower:
        passed = False
        details += " | AI did not ask about willingness to pay or urgency"

    # AI should suggest next test
    has_action = len(resp.actions) > 0

    print_result(
        "Problem validation asks right questions",
        passed and has_action,
        resp.reply,
        details,
    )


# ============================================================
# TEST GROUP 3: Cross-Module Intelligence (BMC references Problem)
# ============================================================

def test_bmc_references_problem_statement():
    """On BMC page, AI should reference problem statement to suggest customer segments."""
    agent = get_conversational_agent()

    # Scenario: User is on BMC, says customer segment is "schools" with problem context
    req = make_request(
        "business",
        "Mon customer segment c'est les ecoles",
        filled_fields=[
            {"field_name": "problemStatement", "label": "Enonce", "is_filled": True,
             "content": "Les eleves a Dakar perdent 2h par jour sur les reseaux sociaux au lieu d'etudier"},
            {"field_name": "customerSegments", "label": "Segments clients", "is_filled": True, "content": "Les ecoles"},
        ],
        empty_fields=["valuePropositions", "revenueStreams", "channels"],
    )
    resp = agent.run(req)

    passed = True
    details = ""
    reply_lower = resp.reply.lower()

    # AI should suggest more specific segment based on typical problems
    if "eleve" not in reply_lower and "etudiant" not in reply_lower and "age" not in reply_lower and "profil" not in reply_lower and "cible" not in reply_lower:
        passed = False
        details = "AI did not suggest a more specific segment"

    # AI should give reasoning
    if "parce" not in reply_lower and "car" not in reply_lower and "donc" not in reply_lower and "raison" not in reply_lower and "pourquoi" not in reply_lower:
        passed = False
        details += " | AI did not provide reasoning"

    print_result(
        "BMC suggests specific customer segment with reasoning",
        passed,
        resp.reply,
        details,
    )


# ============================================================
# TEST GROUP 4: ICP Cross-Referencing
# ============================================================

def test_icp_references_problem_and_bmc():
    """On ICP page, AI should reference problem statement data."""
    agent = get_conversational_agent()

    req = make_request(
        "icp",
        "Mon ICP c'est les jeunes au Senegal",
        filled_fields=[
            {"field_name": "icpDescription", "label": "Description ICP", "is_filled": True, "content": "Les jeunes au Senegal"},
        ],
        empty_fields=["personaNarrative", "jtbd", "buyingContext"],
    )
    resp = agent.run(req)

    passed = True
    details = ""
    reply_lower = resp.reply.lower()

    # AI should challenge broadness
    if "large" not in reply_lower and "vague" not in reply_lower and "precis" not in reply_lower and "specific" not in reply_lower and "flou" not in reply_lower:
        passed = False
        details = "AI did not flag ICP as too broad"

    # AI should suggest specifics
    if "age" not in reply_lower and "tranche" not in reply_lower and "13" not in reply_lower and "20" not in reply_lower:
        passed = False
        details += " | AI did not suggest age range or specific criteria"

    print_result(
        "ICP challenges broad segment and suggests specifics",
        passed,
        resp.reply,
        details,
    )


# ============================================================
# TEST GROUP 5: Multi-Step Conversation Chain
# ============================================================

def test_full_validation_chain():
    """Full chain: vague problem → challenge → reformulate → validate → cross-module."""
    agent = get_conversational_agent()
    responses = []

    # Step 1: Vague problem
    req1 = make_request(
        "problem-statement",
        "Au Senegal les reseaux sociaux c'est un probleme"
    )
    resp1 = agent.run(req1)
    responses.append(("vague_problem", resp1))

    # Step 2: User asks to focus on students
    req2 = make_request(
        "problem-statement",
        "Reformule avec un focus sur les eleves de Dakar"
    )
    resp2 = agent.run(req2)
    responses.append(("reformulate_students", resp2))

    # Step 3: Move to validation
    req3 = make_request(
        "problem-validation",
        "Les eleves de 13 a 20 ans a Dakar passent en moyenne 4h par jour sur TikTok et Instagram au lieu de faire leurs devoirs. 7 eleves sur 10 disent que ca affecte leurs notes.",
        filled_fields=[
            {"field_name": "problemStatement", "label": "Enonce", "is_filled": True,
             "content": "Eleves 13-20 ans Dakar: 4h/jour reseaux sociaux au lieu d'etudier"},
        ],
        empty_fields=["evidence", "interviewsCount"],
    )
    resp3 = agent.run(req3)
    responses.append(("validation_detailed", resp3))

    # Step 4: Move to ICP
    req4 = make_request(
        "icp",
        "Mon ICP: eleves lycee a Dakar",
        filled_fields=[
            {"field_name": "icpDescription", "label": "Description ICP", "is_filled": True, "content": "Eleves lycee a Dakar"},
        ],
        empty_fields=["personaNarrative", "jtbd"],
    )
    resp4 = agent.run(req4)
    responses.append(("icp_refined", resp4))

    # Step 5: Move to BMC
    req5 = make_request(
        "business",
        "Mon BMC: customer=eleves, valeur=meilleures notes, revenus=abonnement parents",
        filled_fields=[
            {"field_name": "customerSegments", "label": "Segments", "is_filled": True, "content": "Eleves lycee"},
            {"field_name": "valuePropositions", "label": "Valeur", "is_filled": True, "content": "Meilleures notes"},
            {"field_name": "revenueStreams", "label": "Revenus", "is_filled": True, "content": "Abonnement parents"},
        ],
        empty_fields=["channels", "keyPartners"],
    )
    resp5 = agent.run(req5)
    responses.append(("bmc_coherence", resp5))

    # Validate each step
    all_passed = True
    for name, resp in responses:
        has_content = len(resp.reply) > 30
        has_action = len(resp.actions) > 0
        has_knowledge = len(resp.supporting_context) > 0
        step_passed = has_content and has_action
        if not step_passed:
            all_passed = False
        print_result(
            f"Chain step: {name}",
            step_passed,
            resp.reply,
            f"content={has_content}, actions={has_action}, knowledge={has_knowledge}",
        )

    return all_passed


# ============================================================
# TEST GROUP 6: Specific Problematic Scenarios (Senegal-based)
# ============================================================

def test_senegal_specific_problematics():
    """Test various Senegal-based problem statements."""
    agent = get_conversational_agent()

    scenarios = [
        {
            "name": "transport_dakar",
            "module": "problem-statement",
            "message": "A Dakar les transports en commun sont un probleme",
            "expect_keywords": ["vague", "qui", "precis", "flou"],
        },
        {
            "name": "chomage_jeunes",
            "module": "problem-statement",
            "message": "Le chomage des jeunes au Senegal est tres eleve",
            "expect_keywords": ["vague", "qui", "precis", "chiffre", "donnee"],
        },
        {
            "name": "sante_rurale",
            "module": "problem-statement",
            "message": "Les villages ruraux n'ont pas acces aux soins de sante",
            "expect_keywords": ["precis", "quel", "village", "region", "distance"],
        },
        {
            "name": "education_numerique",
            "module": "problem-statement",
            "message": "Les eleves du secondaire n'ont pas acces a l'education numerique de qualite",
            "expect_keywords": ["eleve", "secondaire", "acces", "qualite"],
        },
    ]

    all_passed = True
    for scenario in scenarios:
        req = make_request(scenario["module"], scenario["message"])
        resp = agent.run(req)

        reply_lower = resp.reply.lower()
        found_keywords = [kw for kw in scenario["expect_keywords"] if kw in reply_lower]

        passed = len(found_keywords) >= 2
        if not passed:
            all_passed = False

        print_result(
            f"Senegal scenario: {scenario['name']}",
            passed,
            resp.reply,
            f"Found {len(found_keywords)}/{len(scenario['expect_keywords'])} expected keywords: {found_keywords}",
        )

    return all_passed


# ============================================================
# TEST GROUP 7: Problem Validation Questions Flow
# ============================================================

def test_validation_question_flow():
    """Test that AI asks the right validation questions in sequence."""
    agent = get_conversational_agent()

    validation_stages = [
        {
            "name": "stage_1_no_evidence",
            "module": "problem-validation",
            "filled": [],
            "empty": ["evidence", "interviewsCount", "willingnessToPay"],
            "message": "Je pense que mon probleme est valide",
            "expect": ["interview", "preuve", "evidence", "temoin", "terrain"],
        },
        {
            "name": "stage_2_weak_evidence",
            "module": "problem-validation",
            "filled": [
                {"field_name": "evidence", "label": "Preuves", "is_filled": True,
                 "content": "J'ai demande a 2 amis"},
            ],
            "empty": ["interviewsCount", "willingnessToPay"],
            "message": "J'ai parle a 2 amis qui sont d'accord avec moi",
            "expect": ["amis", "biais", "famille", "objectif", "terrain", "client"],
        },
        {
            "name": "stage_3_good_evidence",
            "module": "problem-validation",
            "filled": [
                {"field_name": "evidence", "label": "Preuves", "is_filled": True,
                 "content": "10 interviews, 8 confirment le probleme, 6 paient deja pour une solution"},
            ],
            "empty": [],
            "message": "10 interviews: 8 confirment le probleme, 6 paient deja 5000 FCFA pour un outil similaire",
            "expect": ["valide", "solide", "prochain", "etape", "bmc", "icp", "solution"],
        },
    ]

    all_passed = True
    for stage in validation_stages:
        req = make_request(
            stage["module"],
            stage["message"],
            filled_fields=stage["filled"],
            empty_fields=stage["empty"],
        )
        resp = agent.run(req)

        reply_lower = resp.reply.lower()
        found = [kw for kw in stage["expect"] if kw in reply_lower]

        passed = len(found) >= 1
        if not passed:
            all_passed = False

        print_result(
            f"Validation flow: {stage['name']}",
            passed,
            resp.reply,
            f"Found {len(found)}/{len(stage['expect'])} keywords: {found}",
        )

    return all_passed


# ============================================================
# TEST GROUP 8: Cross-Module Coherence Checks
# ============================================================

def test_cross_module_coherence():
    """Test that AI catches incoherence between modules."""
    agent = get_conversational_agent()

    coherence_tests = [
        {
            "name": "bmc_vs_problem_mismatch",
            "module": "business",
            "message": "Mon probleme c'est les eleves et les reseaux sociaux, mais mon customer segment c'est les entreprises",
            "expect": ["eleve", "incoher", "contradict", "align", "coh", "segment"],
        },
        {
            "name": "gtm_without_icp",
            "module": "gtm",
            "message": "Je veux lancer ma campagne marketing sur Facebook et Instagram",
            "expect": ["icp", "client", "cible", "qui", "savoir"],
        },
        {
            "name": "roi_without_market_sizing",
            "module": "roi",
            "message": "Mon ROI sera de 500% en 6 mois",
            "expect": ["hypoth", "base", "chiffre", "donnee", "source", "justif"],
        },
    ]

    all_passed = True
    for test in coherence_tests:
        req = make_request(test["module"], test["message"])
        resp = agent.run(req)

        reply_lower = resp.reply.lower()
        found = [kw for kw in test["expect"] if kw in reply_lower]

        passed = len(found) >= 2
        if not passed:
            all_passed = False

        print_result(
            f"Cross-module coherence: {test['name']}",
            passed,
            resp.reply,
            f"Found {len(found)}/{len(test['expect'])} keywords: {found}",
        )

    return all_passed


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("QWEN3 FINE-TUNED MODEL - ADVANCED TEST SUITE")
    print("=" * 70)
    print()

    results = {}

    print("--- GROUP 1: Vague Problem -> Challenge -> Reformulate ---")
    test_vague_problem_challenge_reformulate()
    print()

    print("--- GROUP 2: Problem Validation - AI Knows Questions ---")
    test_problem_validation_knows_questions()
    print()

    print("--- GROUP 3: BMC References Problem Statement ---")
    test_bmc_references_problem_statement()
    print()

    print("--- GROUP 4: ICP Cross-Referencing ---")
    test_icp_references_problem_and_bmc()
    print()

    print("--- GROUP 5: Full Validation Chain ---")
    test_full_validation_chain()
    print()

    print("--- GROUP 6: Senegal-Specific Problematics ---")
    test_senegal_specific_problematics()
    print()

    print("--- GROUP 7: Validation Question Flow ---")
    test_validation_question_flow()
    print()

    print("--- GROUP 8: Cross-Module Coherence ---")
    test_cross_module_coherence()
    print()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
