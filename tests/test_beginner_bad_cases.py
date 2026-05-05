r"""
Behavioral unit tests for beginner / messy user prompts from problem statement to GTM.

This file is pytest-compatible, and it can also be run directly:
    .venv\Scripts\python.exe tests\test_beginner_bad_cases.py
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.conversational import ConversationalAgent
from app.schemas.chat import ChatMessageItem, ChatRequest, ModuleContext
from app.services.founderpath.client import FounderPathClient
from app.services.retrieval.in_memory import InMemoryRetriever
from app.tools.project_snapshot import ProjectSnapshotTool


def make_request(
    module_key: str,
    message: str,
    *,
    locale: str = "fr",
    filled_fields: list | None = None,
    empty_fields: list | None = None,
    conversation_history: list | None = None,
) -> ChatRequest:
    return ChatRequest(
        module=ModuleContext(
            module_key=module_key,
            label=module_key,
            filled_fields=filled_fields or [],
            empty_fields=empty_fields or [],
        ),
        message=message,
        locale=locale,
        conversation_history=conversation_history or [],
    )


@lru_cache(maxsize=1)
def get_test_agent() -> ConversationalAgent:
    return ConversationalAgent(
        project_snapshot_tool=ProjectSnapshotTool(FounderPathClient()),
        retriever=InMemoryRetriever(),
    )


def assert_any(text: str, candidates: list[str], message: str) -> None:
    lowered = text.lower()
    if not any(candidate in lowered for candidate in candidates):
        raise AssertionError(f"{message}. Missing any of: {candidates}\nReply was:\n{text}")


def assert_action_type(response, action_type: str) -> None:
    if not any(action.type == action_type for action in response.actions):
        raise AssertionError(f"Expected action type '{action_type}' but got {[action.type for action in response.actions]}")


def test_problem_badly_written_should_be_challenged():
    agent = get_test_agent()
    response = agent.run(make_request("problem-statement", "g un pb reseau sociaux senegal"))

    assert_any(response.reply, ["vague", "flou", "precise", "plus large"], "The model should challenge a messy vague problem")
    assert_any(response.reply, ["qui", "quand", "combien", "impact"], "The model should ask for missing problem details")
    assert_action_type(response, "quick_prompt")


def test_problem_question_should_be_reframed_as_statement():
    agent = get_test_agent()
    response = agent.run(
        make_request(
            "problem-statement",
            "Comment les mamans a Dakar peuvent mieux gerer le temps d ecran de leurs enfants ?",
        )
    )

    assert_any(response.reply, ["question", "affirmation", "douleur concrete"], "The model should explain that a question is not a strong problem statement")
    assert_any(response.reply, ["version", "proposee", "plus clair"], "The model should still help reformulate")


def test_problem_reformulation_should_use_history_and_user_hint():
    agent = get_test_agent()
    history = [ChatMessageItem(role="user", content="Au Senegal on a un probleme avec les reseaux sociaux")]
    response = agent.run(
        make_request(
            "problem-statement",
            "reformule stp, axe sur les eleves de Dakar",
            conversation_history=history,
        )
    )

    assert_any(response.reply, ["eleves", "etude", "reseaux sociaux"], "The reformulation should use the student hint and original topic")
    assert_any(response.reply, ["dakar", "senegal"], "The reformulation should keep a local context")
    if "reformule stp" in response.reply.lower():
        raise AssertionError(f"The model echoed the instruction instead of reformulating the underlying problem.\nReply was:\n{response.reply}")
    assert_action_type(response, "apply_fields")


def test_fintech_problem_reformulation_should_use_business_context():
    agent = get_test_agent()
    history = [ChatMessageItem(role="user", content="les PME ont des problemes de tresorerie")]
    response = agent.run(
        make_request(
            "problem-statement",
            "reformule stp pour des grossistes a Dakar",
            conversation_history=history,
        )
    )

    assert_any(response.reply, ["grossistes", "dakar"], "The reformulation should use the requested audience and location")
    assert_any(response.reply, ["tresorerie", "paient en retard", "stocks"], "The reformulation should make the treasury problem more concrete")
    assert_action_type(response, "apply_fields")


def test_problem_validation_should_flag_friend_bias():
    agent = get_test_agent()
    response = agent.run(
        make_request(
            "problem-validation",
            "Mes amis disent que mon idee est top donc c est valide",
            filled_fields=[
                {"field_name": "evidence", "label": "Preuves", "is_filled": True, "content": "3 amis aiment bien l idee"}
            ],
            empty_fields=["interviewsCount", "willingnessToPay"],
        )
    )

    assert_any(response.reply, ["biais", "amis", "famille", "clients potentiels reels"], "The model should reject weak evidence from friends")
    assert_action_type(response, "quick_prompt")


def test_research_should_help_extract_signals_from_noisy_feedback():
    agent = get_test_agent()
    response = agent.run(
        make_request(
            "research",
            "j ai des retours un peu fouillis, aide moi a sortir les vrais signaux",
            filled_fields=[
                {
                    "field_name": "transcription",
                    "label": "Transcription",
                    "is_filled": True,
                    "content": "Franchement je passe trop de temps dessus, ca m enerve, je paierais peut etre si c etait simple.",
                }
            ],
            empty_fields=["keyEvidence"],
        )
    )

    assert_any(response.reply, ["pain", "willingness to pay", "buying signals", "objections", "signaux forts"], "The model should structure interview signal extraction")
    assert_action_type(response, "quick_prompt")


def test_icp_should_reject_overly_broad_beginner_segment():
    agent = get_test_agent()
    response = agent.run(make_request("icp", "Mon ICP c est les entrepreneurs en Afrique"))

    assert_any(response.reply, ["trop large", "personne en particulier", "version de depart plus actionnable"], "The model should reject a continent-sized ICP")
    assert_any(response.reply, ["profil precis", "ville", "qui decide", "sous-segment"], "The model should force a more operational segmentation")


def test_b2b_icp_should_not_fall_back_to_student_template():
    agent = get_test_agent()
    response = agent.run(make_request("icp", "Mon ICP c est les entreprises en Afrique"))

    assert_any(response.reply, ["b2b", "grossistes", "distribution", "retards de paiement"], "The model should propose a business-relevant narrowing example")
    if "eleves" in response.reply.lower() or "parents d'eleves" in response.reply.lower():
        raise AssertionError(f"The model fell back to a student-specific ICP template.\nReply was:\n{response.reply}")


def test_business_should_catch_segment_mismatch_with_problem():
    agent = get_test_agent()
    response = agent.run(
        make_request(
            "business",
            "Mon customer segment c est les entreprises",
            filled_fields=[
                {
                    "field_name": "problemStatement",
                    "label": "Enonce",
                    "is_filled": True,
                    "content": "Les eleves a Dakar perdent 4h par jour sur les reseaux sociaux au lieu d etudier",
                }
            ],
            empty_fields=["valuePropositions", "revenueStreams"],
        )
    )

    assert_any(response.reply, ["entreprises", "eleves", "parents", "meilleur customer segment"], "The model should catch a segment mismatch")
    assert_any(response.reply, ["pourquoi", "decision", "budget"], "The model should justify the correction")


def test_competition_should_reject_no_competitor_claim():
    agent = get_test_agent()
    response = agent.run(make_request("competitive-landscape", "Je n ai pas de concurrents"))

    assert_any(response.reply, ["4 types", "status quo", "indirect", "budget competitors"], "The model should explain why no-competition claims are naive")
    assert_action_type(response, "quick_prompt")


def test_market_sizing_should_block_before_problem_validation():
    agent = get_test_agent()
    response = agent.run(make_request("market-sizing", "Comment calculer mon TAM avant de valider mon probleme ?"))

    assert_any(response.reply, ["tu vas trop vite", "valider le probleme", "interviews", "ordre recommande"], "The model should push validation before sizing")
    assert_action_type(response, "quick_prompt")


def test_market_sizing_should_reject_top_down_fantasy():
    agent = get_test_agent()
    response = agent.run(make_request("market-sizing", "Le marche africain vaut des milliards donc mon TAM est enorme"))

    assert_any(response.reply, ["attention au top-down", "bottom-up", "tam", "sam", "som"], "The model should redirect from fantasy TAM to bottom-up sizing")
    assert_action_type(response, "quick_prompt")


def test_gtm_should_reject_feature_first_launch_plan():
    agent = get_test_agent()
    response = agent.run(make_request("gtm", "Mon produit a 10 fonctionnalites et mon app est presque finie, comment le lancer ?"))

    assert_any(response.reply, ["produit, pas par le marche", "qui souffre", "promesse simple", "segment"], "The model should stop a feature-first GTM")
    assert_action_type(response, "quick_prompt")


def test_gtm_should_not_choose_channels_before_icp():
    agent = get_test_agent()
    response = agent.run(make_request("gtm", "Je veux lancer une campagne Facebook et Instagram demain"))

    assert_any(response.reply, ["qui tu cibles", "icp", "parents", "eleves"], "The model should route channel choice through ICP")
    assert_action_type(response, "quick_prompt")


def test_sensitive_health_problem_should_be_challenged_more_strictly():
    agent = get_test_agent()
    response = agent.run(make_request("problem-statement", "les femmes enceintes font parfois de la depression"))

    assert_any(response.reply, ["sante sensible", "signes observables", "quasi-diagnostiques", "impact sur la vie quotidienne"], "The model should add extra caution on sensitive health problems")
    assert_any(response.reply, ["detresse emotionnelle", "grossesse", "suivi"], "The model should offer a more operational rewrite for the health case")


def test_sensitive_health_icp_should_reject_continent_scale_segment():
    agent = get_test_agent()
    response = agent.run(make_request("icp", "Mon ICP c est les femmes enceintes en Afrique"))

    assert_any(response.reply, ["trop large", "maternites precises", "stade de grossesse", "recommande"], "The model should challenge a broad maternal-health ICP")
    assert_action_type(response, "quick_prompt")


def test_sensitive_health_business_should_reject_generic_companies_segment():
    agent = get_test_agent()
    response = agent.run(
        make_request(
            "business",
            "Mon customer segment c est les entreprises",
            filled_fields=[
                {
                    "field_name": "problemStatement",
                    "label": "Enonce",
                    "is_filled": True,
                    "content": "Les femmes enceintes vivent des episodes de detresse emotionnelle pendant la grossesse et n osent pas en parler",
                }
            ],
            empty_fields=["valuePropositions", "revenueStreams"],
        )
    )

    assert_any(response.reply, ["grossesse", "detresse", "pas le meilleur customer segment", "maternites"], "The model should reject a generic B2B segment for the maternal-health case")
    assert_any(response.reply, ["confiance", "recommande", "plus sure"], "The model should justify with trust and field-learning logic")


def run_manual_suite() -> int:
    tests = [
        test_problem_badly_written_should_be_challenged,
        test_problem_question_should_be_reframed_as_statement,
        test_problem_reformulation_should_use_history_and_user_hint,
        test_fintech_problem_reformulation_should_use_business_context,
        test_problem_validation_should_flag_friend_bias,
        test_research_should_help_extract_signals_from_noisy_feedback,
        test_icp_should_reject_overly_broad_beginner_segment,
        test_b2b_icp_should_not_fall_back_to_student_template,
        test_business_should_catch_segment_mismatch_with_problem,
        test_competition_should_reject_no_competitor_claim,
        test_market_sizing_should_block_before_problem_validation,
        test_market_sizing_should_reject_top_down_fantasy,
        test_gtm_should_reject_feature_first_launch_plan,
        test_gtm_should_not_choose_channels_before_icp,
        test_sensitive_health_problem_should_be_challenged_more_strictly,
        test_sensitive_health_icp_should_reject_continent_scale_segment,
        test_sensitive_health_business_should_reject_generic_companies_segment,
    ]

    failures = 0
    print("=" * 72)
    print("BEGINNER BAD CASES - MANUAL TEST SUITE")
    print("=" * 72)

    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")

    print("-" * 72)
    print(f"Total: {len(tests)} | Passed: {len(tests) - failures} | Failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_manual_suite())
