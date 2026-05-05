import pytest
import json
from app.schemas.chat import ChatRequest, ModuleContext, ChatAction
from app.agents.conversational import ConversationalAgent
from app.registry import get_conversational_agent, get_retriever
from app.services.retrieval.in_memory import InMemoryRetriever


@pytest.fixture
def agent():
    return get_conversational_agent()


@pytest.fixture
def retriever():
    return get_retriever()


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


# ============================================================
# MODULE 1: PROBLEM STATEMENT
# ============================================================

class TestProblemStatement:
    def test_vague_problem_should_flag_all_missing(self, agent):
        """User gives a very vague 3-word problem - should flag everything missing"""
        req = make_request("problem-statement", "C'est un probleme")
        resp = agent.run(req)

        assert "trop vague" in resp.reply.lower() or "vague" in resp.reply.lower()
        assert "qui" in resp.reply.lower()
        assert len(resp.actions) > 0

    def test_solution_oriented_problem_should_warn(self, agent):
        """User talks about solution instead of problem - should warn"""
        req = make_request(
            "problem-statement",
            "Je veux creer une application mobile pour aider les commercants"
        )
        resp = agent.run(req)

        assert "solution" in resp.reply.lower()
        assert any("solution" in r.lower() for r in resp.reply.split("\n"))

    def test_question_format_should_reject(self, agent):
        """User phrases problem as a question - should reject"""
        req = make_request(
            "problem-statement",
            "Comment les femmes entrepreneurs peuvent-elles mieux gerer leur temps?"
        )
        resp = agent.run(req)

        assert "question" in resp.reply.lower() or "affirmation" in resp.reply.lower()

    def test_good_problem_with_missing_frequency(self, agent):
        """Good problem statement but missing frequency - should flag frequency or solution orientation"""
        req = make_request(
            "problem-statement",
            "Les restaurateurs a Dakar perdent 50 000 FCFA par mois car ils n ont pas de systeme de suivi des commandes"
        )
        resp = agent.run(req)

        assert "frequence" in resp.reply.lower() or "solution" in resp.reply.lower() or "impact" in resp.reply.lower()
        assert "version" in resp.reply.lower() or "proposee" in resp.reply.lower() or "propose" in resp.reply.lower()

    def test_complete_problem_should_generate_apply_action(self, agent):
        """Complete problem should generate apply_fields action with field proposals"""
        req = make_request(
            "problem-statement",
            "Les commercants de March Sandaga a Dakar perdent 3 heures par jour a cause de la gestion manuelle de leurs stocks sur papier"
        )
        resp = agent.run(req)

        apply_actions = [a for a in resp.actions if a.type == "apply_fields"]
        assert len(apply_actions) > 0
        assert len(apply_actions[0].field_proposals) > 0

    def test_problem_extraction_identifies_who(self, agent):
        """Problem with clear target - should extract who field"""
        req = make_request(
            "problem-statement",
            "Les femmes entrepreneurs a Dakar ont du mal a trouver des financements"
        )
        resp = agent.run(req)

        proposals = []
        for action in resp.actions:
            if action.type == "apply_fields":
                proposals.extend(action.field_proposals)

        who_proposals = [p for p in proposals if p.field_name == "who"]
        assert len(who_proposals) > 0
        assert "femmes" in who_proposals[0].value.lower() or "entrepreneurs" in who_proposals[0].value.lower()

    def test_problem_extraction_identifies_cost(self, agent):
        """Problem with cost indicator - should extract cost field"""
        req = make_request(
            "problem-statement",
            "Les PME perdent 200 000 FCFA chaque mois a cause des erreurs de facturation"
        )
        resp = agent.run(req)

        proposals = []
        for action in resp.actions:
            if action.type == "apply_fields":
                proposals.extend(action.field_proposals)

        cost_proposals = [p for p in proposals if p.field_name == "cost"]
        assert len(cost_proposals) > 0 or "cout" in resp.reply.lower() or "impact" in resp.reply.lower() or "prix" in resp.reply.lower()

    def test_problem_knowledge_retrieval(self, agent):
        """Problem module should return relevant knowledge chunks"""
        req = make_request(
            "problem-statement",
            "Comment formuler un bon probleme?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("validation" in t.lower() or "probleme" in t.lower() for t in titles)

    def test_empty_problem_should_prompt_for_help(self, agent):
        """User asks for help with empty fields - should get quick_prompt action"""
        req = make_request(
            "problem-statement",
            "Aide-moi a formuler mon probleme",
            filled_fields=[],
            empty_fields=["who", "problemStatement", "cost"]
        )
        resp = agent.run(req)

        quick_prompts = [a for a in resp.actions if a.type == "quick_prompt"]
        assert len(quick_prompts) > 0


# ============================================================
# MODULE 2: PROBLEM VALIDATION
# ============================================================

class TestProblemValidation:
    def test_validation_with_sufficient_evidence(self, agent):
        """User presents validated evidence - should acknowledge"""
        req = make_request(
            "problem-validation",
            "J'ai interviewe 10 commercants. 8 sur 10 ont confirme qu'ils perdent 2h par jour. 6 paient deja 5000 FCFA pour un outil.",
            filled_fields=[
                {"field_name": "evidence", "label": "Preuves", "is_filled": True, "content": "8/10 perdent 2h/jour"}
            ],
            empty_fields=[]
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0

    def test_validation_with_weak_evidence(self, agent):
        """User presents weak evidence (opinions) - should flag"""
        req = make_request(
            "problem-validation",
            "J'ai demande a 3 amis s'ils achèteraient mon produit. Ils ont dit oui."
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0

    def test_validation_mentions_solution_should_flag(self, agent):
        """User mentions their solution in validation questions - should flag as error"""
        req = make_request(
            "problem-validation",
            "J'ai demande aux gens: Est-ce que vous achèteriez mon application de gestion de stock a 5000 FCFA?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0

    def test_validation_asks_for_next_test(self, agent):
        """User asks what test to do next - should provide guidance"""
        req = make_request(
            "problem-validation",
            "Quel est le prochain test que je devrais faire?"
        )
        resp = agent.run(req)

        assert len(resp.reply) > 50
        assert len(resp.actions) > 0


# ============================================================
# MODULE 3: RESEARCH / INTERVIEWS
# ============================================================

class TestResearch:
    def test_extract_strong_signals(self, agent):
        """User asks to extract strong signals from interviews"""
        req = make_request(
            "research",
            "Quels sont les signaux forts dans mes interviews?",
            filled_fields=[
                {"field_name": "transcription", "label": "Transcription", "is_filled": True, "content": "Je passe 3h chaque soir a reconcilier mes ventes avec mon cahier. C'est frustrant."}
            ],
            empty_fields=["keyEvidence"]
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0

    def test_prepare_next_questions(self, agent):
        """User asks what questions to ask next"""
        req = make_request(
            "research",
            "Quelles questions poser lors de la prochaine interview?"
        )
        resp = agent.run(req)

        assert len(resp.reply) > 50
        assert len(resp.actions) > 0

    def test_identify_useful_quotes(self, agent):
        """User asks for useful quotes from research"""
        req = make_request(
            "research",
            "Quelles citations sont les plus utiles?"
        )
        resp = agent.run(req)

        assert len(resp.reply) > 0


# ============================================================
# MODULE 4: ICP (Ideal Customer Profile)
# ============================================================

class TestICP:
    def test_icp_too_broad_should_flag(self, agent):
        """User defines ICP as 'everyone' - should flag as too broad"""
        req = make_request(
            "icp",
            "Mon ICP c'est les entrepreneurs en Afrique",
            filled_fields=[
                {"field_name": "icpDescription", "label": "Description ICP", "is_filled": True, "content": "Les entrepreneurs en Afrique"}
            ],
            empty_fields=["personaNarrative", "jtbd", "buyingContext"]
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("humaniser" in t.lower() or "six champs" in t.lower() or "observable" in t.lower() for t in titles)

    def test_icp_specific_should_validate(self, agent):
        """User defines specific ICP - should validate and suggest next steps"""
        req = make_request(
            "icp",
            "Femmes entrepreneurs a Dakar, 28-35 ans, vendent en ligne, 2-5 employes, perdent 4h/semaine sur la logistique",
            filled_fields=[
                {"field_name": "icpDescription", "label": "Description ICP", "is_filled": True, "content": "Femmes entrepreneurs a Dakar, 28-35 ans"}
            ],
            empty_fields=["personaNarrative", "jtbd"]
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("humaniser" in t.lower() or "persona" in t.lower() or "jtbd" in t.lower() for t in titles)

    def test_icp_persona_narrative_guidance(self, agent):
        """User asks for persona narrative help - should provide guidance"""
        req = make_request(
            "icp",
            "Comment humaniser ma cible avec un persona narrative?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("humaniser" in t.lower() or "persona" in t.lower() for t in titles)

    def test_icp_jtbd_formula(self, agent):
        """User asks for JTBD help - should provide formula"""
        req = make_request(
            "icp",
            "Comment formuler le Job-To-Be-Done?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0

    def test_icp_scoring_criteria(self, agent):
        """User asks how to score ICP segments"""
        req = make_request(
            "icp",
            "Comment scorer mes segments d'ICP?"
        )
        resp = agent.run(req)

        assert len(resp.reply) > 0


# ============================================================
# MODULE 5: BUSINESS MODEL CANVAS
# ============================================================

class TestBusinessModel:
    def test_bmc_incomplete_should_flag(self, agent):
        """User has empty BMC - should flag missing blocks"""
        req = make_request(
            "business",
            "Est-ce que mon modele est coherent?",
            filled_fields=[],
            empty_fields=["valuePropositions", "customerSegments", "revenueStreams", "channels"]
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("piliers" in t.lower() or "blocs" in t.lower() or "9" in t for t in titles)

    def test_bmc_value_proposition_guidance(self, agent):
        """User asks about value proposition - should provide guidance"""
        req = make_request(
            "business",
            "Comment definir ma proposition de valeur?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("creation" in t.lower() or "valeur" in t.lower() or "piliers" in t.lower() for t in titles)

    def test_bmc_revenue_stream_guidance(self, agent):
        """User asks about revenue streams - should provide guidance"""
        req = make_request(
            "business",
            "Comment definir mes sources de revenus?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0

    def test_bmc_coherence_check(self, agent):
        """User asks for coherence check with some filled fields"""
        req = make_request(
            "business",
            "Analyse mon contenu et dis-moi ce qui est fort et ce qui est faible.",
            filled_fields=[
                {"field_name": "customerSegments", "label": "Segments clients", "is_filled": True, "content": "Commercants informels a Dakar"},
                {"field_name": "valuePropositions", "label": "Propositions de valeur", "is_filled": True, "content": "Outil de gestion de stock simplifie"}
            ],
            empty_fields=["revenueStreams", "channels", "keyPartners"]
        )
        resp = agent.run(req)

        assert len(resp.reply) > 0
        assert len(resp.actions) > 0

    def test_bmc_evidence_based(self, agent):
        """User asks about evidence-based BMC - should guide towards validation"""
        req = make_request(
            "business",
            "Comment passer d'hypotheses a un modele valide?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("preuve" in t.lower() or "evidence" in t.lower() or "valide" in t.lower() for t in titles)


# ============================================================
# MODULE 6: COMPETITIVE LANDSCAPE
# ============================================================

class TestCompetitiveLandscape:
    def test_competitive_no_competitors_should_flag(self, agent):
        """User says 'no competitors' - should flag as dangerous"""
        req = make_request(
            "competitive-landscape",
            "Je n'ai pas de concurrents"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("4 types" in t.lower() or "types" in t.lower() or "status quo" in t.lower() for t in titles)

    def test_competitive_only_direct_competitors(self, agent):
        """User lists only direct competitors - should suggest other types"""
        req = make_request(
            "competitive-landscape",
            "Mes concurrents sont Jumia et Glovo",
            filled_fields=[
                {"field_name": "competitors", "label": "Concurrents", "is_filled": True, "content": "Jumia, Glovo"}
            ],
            empty_fields=["yourStrengths", "differentiation"]
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("types" in t.lower() or "4 types" in t.lower() for t in titles)

    def test_competitive_positioning_formula(self, agent):
        """User asks for positioning help - should provide formula"""
        req = make_request(
            "competitive-landscape",
            "Comment me positionner face aux concurrents?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("positionnement" in t.lower() for t in titles)

    def test_competitive_switching_cost(self, agent):
        """User asks about switching cost - should provide guidance"""
        req = make_request(
            "competitive-landscape",
            "Comment evaluer le cout du changement pour mes clients?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("switching" in t.lower() or "changement" in t.lower() or "friction" in t.lower() for t in titles)

    def test_competitive_status_quo_analysis(self, agent):
        """User asks about status quo competitor"""
        req = make_request(
            "competitive-landscape",
            "Qu'est-ce que mes clients font actuellement sans ma solution?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0

    def test_competitive_ground_validation(self, agent):
        """User asks how to validate competitors with field research"""
        req = make_request(
            "competitive-landscape",
            "Comment valider mes concurrents avec des donnees terrain?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        assert len(resp.reply) > 0


# ============================================================
# MODULE 7: MARKET SIZING
# ============================================================

class TestMarketSizing:
    def test_market_sizing_top_down_should_flag(self, agent):
        """User uses top-down sizing - should flag and suggest bottom-up"""
        req = make_request(
            "market-sizing",
            "Le marche africain vaut 100 milliards, donc mon TAM = 100 milliards",
            filled_fields=[
                {"field_name": "tam", "label": "TAM", "is_filled": True, "content": "100 milliards"}
            ],
            empty_fields=["sam", "som"]
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("bottom-up" in t.lower() or "methodologie" in t.lower() for t in titles)

    def test_market_sizing_bottom_up_example(self, agent):
        """User asks for bottom-up methodology - should provide guidance"""
        req = make_request(
            "market-sizing",
            "Comment calculer TAM SAM SOM correctement avec la methode bottom-up?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("bottom-up" in t.lower() or "tam" in t.lower() for t in titles)

    def test_market_sizing_overly_optimistic_som(self, agent):
        """User claims 20% SOM in year 1 - should flag as unrealistic"""
        req = make_request(
            "market-sizing",
            "Mon SOM c'est 20% du SAM en annee 1"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("som" in t.lower() or "realistic" in t.lower() for t in titles)

    def test_market_sizing_validation_required(self, agent):
        """User asks about market sizing before problem validation - should flag"""
        req = make_request(
            "market-sizing",
            "Comment calculer mon TAM avant de valider mon probleme?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("validation" in t.lower() for t in titles)

    def test_market_sizing_filters_for_sam(self, agent):
        """User asks about SAM filters - should provide guidance"""
        req = make_request(
            "market-sizing",
            "Comment definir mes filtres pour le SAM?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("sam" in t.lower() or "filtres" in t.lower() for t in titles)

    def test_market_sizing_sources_citation(self, agent):
        """User asks about citing sources - should emphasize importance"""
        req = make_request(
            "market-sizing",
            "Quelles sources dois-je citer pour mes chiffres?"
        )
        resp = agent.run(req)

        assert len(resp.reply) > 0


# ============================================================
# MODULE 8: GO-TO-MARKET
# ============================================================

class TestGTM:
    def test_gtm_icp_scoring(self, agent):
        """User asks about ICP scoring for GTM - should provide 4-criteria method"""
        req = make_request(
            "gtm",
            "Comment scorer mes segments ICP pour choisir le bon canal?",
            filled_fields=[],
            empty_fields=["channels", "messaging", "icpPrimary"]
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("icp" in t.lower() or "scoring" in t.lower() or "primaire" in t.lower() for t in titles)

    def test_gtm_validation_infrastructure(self, agent):
        """User asks about validation infrastructure - should provide CRM + script guidance"""
        req = make_request(
            "gtm",
            "Comment preparer mon infrastructure de validation?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("validation" in t.lower() or "crm" in t.lower() or "infrastructure" in t.lower() for t in titles)

    def test_gtm_reverse_construction_warning(self, agent):
        """User starts with product instead of customer - should warn"""
        req = make_request(
            "gtm",
            "Mon produit a 10 fonctionnalites. Comment le lancer?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("envers" in t.lower() or "reverse" in t.lower() or "customer" in t.lower() or "qui" in t.lower() for t in titles)

    def test_gtm_daci_framework(self, agent):
        """User asks about decision framework - should provide guidance"""
        req = make_request(
            "gtm",
            "Comment organiser les decisions dans mon equipe GTM?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        assert len(resp.reply) > 0

    def test_gtm_30_day_plan(self, agent):
        """User asks for 30-day launch plan - should provide structured plan"""
        req = make_request(
            "gtm",
            "Genere un plan de lancement simple sur 30 jours."
        )
        resp = agent.run(req)

        assert len(resp.reply) > 50
        assert len(resp.actions) > 0

    def test_gtm_outreach_guidance(self, agent):
        """User asks about outreach strategy - should provide guidance"""
        req = make_request(
            "gtm",
            "Comment faire mon outreach aux 10 premiers contacts?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("outreach" in t.lower() or "validation" in t.lower() or "contact" in t.lower() for t in titles)

    def test_gtm_value_proof_points(self, agent):
        """User asks about value proof points - should provide guidance"""
        req = make_request(
            "gtm",
            "Quelles preuves de valeur donner a mes premiers clients?"
        )
        resp = agent.run(req)

        assert len(resp.supporting_context) > 0
        titles = [c.title for c in resp.supporting_context]
        assert any("preuve" in t.lower() or "valeur" in t.lower() or "envers" in t.lower() for t in titles)


# ============================================================
# CROSS-MODULE TESTS
# ============================================================

class TestCrossModule:
    def test_all_modules_return_knowledge(self, agent):
        """All modules should return at least some knowledge context"""
        modules = [
            "problem-statement", "problem-validation", "research",
            "icp", "business", "competitive-landscape", "market-sizing", "gtm"
        ]

        for module in modules:
            req = make_request(module, "Aide-moi sur cette page")
            resp = agent.run(req)

            assert len(resp.supporting_context) > 0, f"Module {module} returned no knowledge context"
            assert len(resp.reply) > 0, f"Module {module} returned empty reply"

    def test_problem_statement_returns_more_chunks(self, agent):
        """Problem-statement module should have the most knowledge chunks"""
        req = make_request("problem-statement", "Aide-moi")
        resp = agent.run(req)

        assert len(resp.supporting_context) >= 3

    def test_knowledge_relevance_by_module(self, retriever):
        """Each module should return module-specific knowledge, not generic"""
        module_queries = {
            "problem-statement": "probleme",
            "icp": "client ideal",
            "business": "business model",
            "competitive-landscape": "concurrents",
            "market-sizing": "tam sam som",
            "gtm": "go-to-market",
            "roi": "roi calcul",
            "journey": "parcours",
        }

        for module, query in module_queries.items():
            results = retriever.search(query=query, module=module, limit=4)

            if retriever.count() > 0:
                assert len(results) > 0, f"No knowledge found for module {module}"

    def test_heuristic_fallback_works(self, agent):
        """Even without LLM, heuristic fallback should provide useful response"""
        req = make_request("gtm", "Aide-moi")
        resp = agent.run(req)

        assert len(resp.reply) > 50
        assert len(resp.actions) > 0


# ============================================================
# RETRIEVER TESTS
# ============================================================

class TestRetriever:
    def test_retriever_loads_all_knowledge(self, retriever):
        """Retriever should load knowledge from all modules"""
        assert retriever.count() >= 30, f"Only {retriever.count()} chunks loaded, expected >= 30"

    def test_retriever_search_by_module(self, retriever):
        """Search should return module-specific results"""
        results = retriever.search(query="test", module="market-sizing", limit=4)
        assert len(results) > 0, "No results for market-sizing module"

    def test_retriever_fallback_for_unknown_module(self, retriever):
        """Unknown module should return empty list, not crash"""
        results = retriever.search(query="test", module="unknown-module", limit=4)
        assert isinstance(results, list)

    def test_retriever_respects_limit(self, retriever):
        """Search should respect the limit parameter"""
        results = retriever.search(query="test", module="problem-statement", limit=2)
        assert len(results) <= 2


# ============================================================
# USER BEHAVIOR SIMULATION TESTS
# ============================================================

class TestUserBehaviorSimulation:
    def test_scenario_complete_journey_problem_to_gtm(self, agent):
        """Simulate a user going through Problem → Validation → ICP → BMC → Competition → Market → GTM"""
        responses = []

        steps = [
            ("problem-statement", "Les commercants a Dakar ont du mal a gerer leurs stocks car ils utilisent des cahiers papier et perdent 3h par jour"),
            ("problem-validation", "J'ai interviewe 10 commercants. 8 confirment qu'ils perdent 2-3h par jour sur la gestion manuelle."),
            ("research", "Dans mes interviews, les commercants disent qu'ils veulent un outil simple sur WhatsApp"),
            ("icp", "Mon ICP: commercants a Dakar, 25-45 ans, 1-3 employes, utilisent WhatsApp quotidiennement"),
            ("business", "Mon BMC: segments=commercants, valeur=gain de temps, revenus=abonnement 5000 FCFA/mois"),
            ("competitive-landscape", "Mes concurrents: cahiers papier (status quo), Excel (direct), groupes WhatsApp (indirect)"),
            ("market-sizing", "TAM = 500 000 commercants x 5000 FCFA x 12 = 30 milliards. SAM = 50 000 (Dakar) x 5000 x 12 = 3 milliards. SOM = 500 (annee 1) x 5000 x 12 = 30 millions"),
            ("gtm", "Mon plan: CRM avec 10 contacts, script d'interview prepare, outreach via WhatsApp cette semaine"),
        ]

        for module, message in steps:
            req = make_request(module, message)
            resp = agent.run(req)
            responses.append(resp)

            assert len(resp.reply) > 0, f"Empty response for {module}"
            assert len(resp.actions) > 0, f"No actions for {module}"
            assert len(resp.supporting_context) > 0, f"No knowledge for {module}"

    def test_scenario_user_makes_common_mistakes(self, agent):
        """Simulate a user making common mistakes across modules"""
        mistakes = [
            ("problem-statement", "Je veux creer une app de livraison"),
            ("problem-validation", "J'ai demande a ma famille s'ils achèteraient"),
            ("icp", "Tout le monde est mon client"),
            ("competitive-landscape", "Je n'ai pas de concurrents"),
            ("market-sizing", "Le marche mondial vaut 1000 milliards"),
        ]

        for module, message in mistakes:
            req = make_request(module, message)
            resp = agent.run(req)

            assert len(resp.reply) > 0
            assert len(resp.supporting_context) > 0

    def test_scenario_user_asks_for_help_at_each_step(self, agent):
        """Simulate a user asking for help at each module"""
        help_messages = [
            "Aide-moi a ecrire",
            "Comment valider mon probleme?",
            "Quels signaux chercher dans mes interviews?",
            "Comment definir mon client ideal?",
            "Est-ce que mon BMC est coherent?",
            "Comment analyser mes concurrents?",
            "Comment calculer TAM SAM SOM?",
            "Quel canal d'acquisition prioriser?",
        ]

        modules = [
            "problem-statement", "problem-validation", "research",
            "icp", "business", "competitive-landscape", "market-sizing", "gtm"
        ]

        for module, message in zip(modules, help_messages):
            req = make_request(module, message)
            resp = agent.run(req)

            assert len(resp.reply) > 50, f"Response too short for {module}"
            assert len(resp.actions) > 0, f"No actions for {module}"

    def test_scenario_progressive_refinement(self, agent):
        """Simulate a user progressively refining their problem statement"""
        refinements = [
            "C'est un probleme",
            "Les commercants ont un probleme",
            "Les commercants a Dakar ont du mal a gerer leurs stocks",
            "Les commercants a Dakar perdent 3h par jour a gerer leurs stocks sur papier",
            "Les commercants de March Sandaga a Dakar perdent 3h/jour et 50 000 FCFA/mois a cause de la gestion manuelle de leurs stocks sur papier",
        ]

        for i, message in enumerate(refinements):
            req = make_request("problem-statement", message)
            resp = agent.run(req)

            assert len(resp.reply) > 0
            assert len(resp.actions) > 0

            if i >= 3:
                apply_actions = [a for a in resp.actions if a.type == "apply_fields"]
                assert len(apply_actions) > 0, f"No apply action at refinement level {i}"

    def test_scenario_english_user(self, agent):
        """Test that English locale works across modules"""
        modules = [
            "problem-statement", "icp", "business", "gtm"
        ]
        english_messages = [
            "Merchants in Dakar struggle to manage their inventory with paper notebooks",
            "My ICP is women entrepreneurs in Dakar, 28-35, selling online",
            "How to define my revenue streams?",
            "Which acquisition channel should I prioritize?"
        ]

        for module, message in zip(modules, english_messages):
            req = make_request(module, message, locale="en")
            resp = agent.run(req)

            assert len(resp.reply) > 0
            assert len(resp.actions) > 0
            assert len(resp.supporting_context) > 0
