# Conversational Module Eval

- Total cases: 20
- Passed: 17
- Failed: 3

## By Module

| Module | Total | Passed | Failed |
| --- | --- | --- | --- |
| business | 2 | 2 | 0 |
| gtm | 3 | 1 | 2 |
| icp | 2 | 2 | 0 |
| journey | 1 | 1 | 0 |
| market-sizing | 3 | 3 | 0 |
| problem-statement | 2 | 1 | 1 |
| problem-validation | 3 | 3 | 0 |
| research | 2 | 2 | 0 |
| roi | 2 | 2 | 0 |

## Cases

| Case | Module | Pass | Actions | Missing expected | Forbidden hits |
| --- | --- | --- | --- | --- | --- |
| problem_statement_vague_beginner | problem-statement | no | quick_prompt, quick_prompt, quick_prompt | trop large, qui exactement, combien | - |
| problem_statement_solution_first | problem-statement | yes | apply_fields, quick_prompt, quick_prompt, quick_prompt | - | - |
| problem_validation_no_evidence | problem-validation | yes | quick_prompt, quick_prompt | - | - |
| problem_validation_weak_evidence | problem-validation | yes | quick_prompt, quick_prompt | - | - |
| problem_validation_good_evidence | problem-validation | yes | quick_prompt, quick_prompt | - | - |
| icp_too_broad_b2b_geo | icp | yes | quick_prompt | - | - |
| icp_refinement_good_start | icp | yes | quick_prompt, quick_prompt | - | - |
| business_segment_incoherence_students | business | yes | quick_prompt, quick_prompt | - | - |
| business_missing_blocks | business | yes | quick_prompt, quick_prompt | - | - |
| gtm_feature_first | gtm | no | quick_prompt, quick_prompt | produit, petit segment | phase 1 |
| gtm_channel_choice_without_icp | gtm | no | quick_prompt, quick_prompt | avant de choisir tes canaux, whatsapp, valide d'abord ton icp | - |
| gtm_three_phase_plan | gtm | yes | quick_prompt, quick_prompt | - | - |
| market_sizing_before_validation | market-sizing | yes | quick_prompt, quick_prompt | - | - |
| market_sizing_top_down | market-sizing | yes | quick_prompt, quick_prompt | - | - |
| market_sizing_bottom_up | market-sizing | yes | quick_prompt, quick_prompt | - | - |
| roi_unfounded_big_claim | roi | yes | quick_prompt | - | - |
| roi_grounded_framework | roi | yes | quick_prompt | - | - |
| research_signal_extraction | research | yes | quick_prompt | - | - |
| research_interview_framework | research | yes | quick_prompt | - | - |
| journey_baseline_framework | journey | yes | quick_prompt | - | - |

## problem_statement_vague_beginner

**Module**: problem-statement

**Message**: Les gens ont un probleme

**Reply**: Envoie-moi ton probleme en une phrase simple, et je te dirai ce qui bloque puis je te proposerai une meilleure version.

**Actions**: quick_prompt, quick_prompt, quick_prompt

**Expected hits**: -

**Missing expected**: trop large, qui exactement, combien

**Forbidden hits**: -

## problem_statement_solution_first

**Module**: problem-statement

**Message**: Je veux creer une application IA pour aider les commercants.

**Reply**: Tu pointes deja une solution. Reviens d abord sur la douleur du client.
Je vois le point de depart. Voici comment le rendre plus clair.
Le vrai manque maintenant, c est plutot :
- dans quel moment concret le probleme apparait
- a quelle frequence cela se produit
- quel impact reel cela cree

**Actions**: apply_fields, quick_prompt, quick_prompt, quick_prompt

**Expected hits**: solution, douleur

**Missing expected**: -

**Forbidden hits**: -

## problem_validation_no_evidence

**Module**: problem-validation

**Message**: Je n'ai pas encore fait d'interviews. Est-ce que mon probleme est valide ?

**Reply**: **Validation : Aucune evidence terrain**

Tu n'as pas encore de preuves terrain. Avant de dire que ton probleme est valide, il te faut des interviews reels.

**Questions a te poser maintenant :**
- Combien de personnes as-tu interviewees?
- Qu'ont-elles confirme sur le probleme?
- Qu'utilisent-elles comme solution actuelle?

**Prochain test :** Interviewe 5-10 personnes de ta cible cette semaine. Pas des amis ou de la famille. Demande-leur de te decrire leur probleme en detail.

**Actions**: quick_prompt, quick_prompt

**Expected hits**: preuves terrain, interviews reels, 5-10

**Missing expected**: -

**Forbidden hits**: -

## problem_validation_weak_evidence

**Module**: problem-validation

**Message**: J'ai parle a 3 amis et 2 membres de ma famille. Ils disent que c'est une bonne idee.

**Reply**: **Validation : Evidence faible (amis/famille)**

Tu as commence a parler a des gens, mais les preuves sont faibles.

**Attention aux biais :**
- Les amis et la famille disent toujours oui (biais de complaisance)
- Il te faut des clients potentiels reels

**Questions pour aller plus loin :**
- As-tu interviewe des clients potentiels reels (pas des proches)?
- Peux-tu me donner des chiffres concrets (temps, argent, frequence)?
- Ont-ils mentionne une solution actuelle qu'ils paient deja?

**Actions**: quick_prompt, quick_prompt

**Expected hits**: preuves sont faibles, amis, clients potentiels reels

**Missing expected**: -

**Forbidden hits**: -

## problem_validation_good_evidence

**Module**: problem-validation

**Message**: J'ai interviewe 12 commercants et 8 ont confirme qu'ils perdent du temps. 5 paient deja un bricolage.

**Reply**: **Validation : Evidence solide**

Ton probleme est solidement valide !

**Ce qui est fort :**
- 12 interviews confirmees avec des commercants de Sandaga.
- Ils confirment perdre 2 a 3 heures par jour.
- 5 paient deja un service manuel ou un assistant pour suivre le stock.

**Prochaines etapes :**
1. Page ICP : Definir ton client ideal avec precision
2. Page BMC : Construire ton modele autour du prix valide
3. Page GTM : Planifier comment atteindre tes 10 premiers clients

**Actions**: quick_prompt, quick_prompt

**Expected hits**: solidement valide, icp, bmc, gtm

**Missing expected**: -

**Forbidden hits**: -

## icp_too_broad_b2b_geo

**Module**: icp

**Message**: Mon ICP ce sont les PME et commercants en Afrique.

**Reply**: C'est beaucoup trop large. 'Tout le monde' = personne en particulier.

**Pourquoi c'est un probleme :**
- Tu melanges des profils, contextes, budgets et urgences tres differents
- Tu apprendras trop lentement si ta cible est definie par un pays ou un continent entier
- Tu ne peux pas construire un produit pour 'tout le monde'

**Version de depart plus actionnable :**
- un profil precis
- dans une ville ou zone precise
- avec une douleur observable
- dans un moment d'usage clair

**Exemple B2B :** grossistes ou PME de distribution a Dakar qui vendent a credit et subissent des retards de paiement recurrents.

**Questions a trancher maintenant :**
- Quel sous-segment souffre le plus ?
- Quel contexte rend la douleur urgente ?
- Qui decide, qui paie et qui utilise ?


**Actions**: quick_prompt

**Expected hits**: beaucoup trop large, tout le monde, sous-segment

**Missing expected**: -

**Forbidden hits**: -

## icp_refinement_good_start

**Module**: icp

**Message**: Mon ICP ce sont les restaurateurs de quartier a Dakar qui gerent encore leurs commandes sur WhatsApp.

**Reply**: Ton ICP a un bon debut. Voici comment le rendre encore plus actionnable :

- **Segment**: Restaurateurs de quartier a Dakar

**Ce qui manque peut-etre :**
- Age precis (tranche d'age)
- Localisation exacte (ville, quartier)
- Budget maximum pour une solution
- Canal de communication prefer


**Actions**: quick_prompt, quick_prompt

**Expected hits**: bon debut, budget maximum, canal de communication

**Missing expected**: -

**Forbidden hits**: -

## business_segment_incoherence_students

**Module**: business

**Message**: Customer segment: les ecoles

**Reply**: Suivant le probleme que tu as defini (eleves et reseaux sociaux), 'les ecoles' n'est pas le meilleur customer segment. Voici pourquoi :

**POURQUOI 'ECOLES' N'EST PAS OPTIMAL :**
- Le probleme est vecu par les **eleves** a la maison, pas a l'ecole
- Les ecoles ont un processus d'achat long (decisions bureaucratiques)
- Le probleme de temps d'ecran se passe en dehors des heures de cours

**SEGMENT PLUS JUDICIEUX :** Les **parents d'eleves de 13 a 20 ans** a Dakar.

**POURQUOI :**
- Ce sont les parents qui paient pour des solutions educatives
- Ils ressentent la douleur indirectement (notes en baisse, conflits familiaux)
- Ils ont le pouvoir de decision et le budget
- Ils cherchent activement des solutions


**Actions**: quick_prompt, quick_prompt

**Expected hits**: pas optimal, parents d'eleves, pouvoir de decision

**Missing expected**: -

**Forbidden hits**: -

## business_missing_blocks

**Module**: business

**Message**: Aide-moi a remplir mon BMC.

**Reply**: Voici les blocs qui manquent et des pistes pour les remplir :

**Channels (Canaux) :** Comment tu atteins tes clients
**Customer-Relationships :** Comment tu les gardes
**Revenue-Streams :** Abonnement, setup, formation
**Key-Resources :** Equipe tech, infrastructure
**Key-Activities :** Dev produit, acquisition, support
**Key-Partnerships :** Associations, fournisseurs
**Cost-Structure :** Salaires, cloud, marketing


**Actions**: quick_prompt, quick_prompt

**Expected hits**: blocs qui manquent, channels, revenue-streams

**Missing expected**: -

**Forbidden hits**: -

## gtm_feature_first

**Module**: gtm

**Message**: Mon application a beaucoup de fonctionnalites. Comment faire mon lancement ?

**Reply**: Voici un plan GTM en 3 phases :

**PHASE 1 : Pre-Launch (Semaines 1-4)**
- Valider l'interet avec 10-10 contacts
- Demo personnalisee de 15 min
- Feedback sur 3 fonctionnalites cles

**PHASE 2 : Launch (Semaines 5-8)**
- 30 clients payants
- Offre lancement : -30% pendant 3 mois
- Partnership avec associations

**PHASE 3 : Post-Launch (Semaines 9-16)**
- 100 clients, churn < 5%
- Programme referral : 1 mois gratuit par parrainage
- Expansion vers d'autres villes


**Actions**: quick_prompt, quick_prompt

**Expected hits**: client

**Missing expected**: produit, petit segment

**Forbidden hits**: phase 1

## gtm_channel_choice_without_icp

**Module**: gtm

**Message**: Je veux faire des campagnes Facebook et Instagram pour lancer mon produit.

**Reply**: Tu commences par le produit, pas par le marche. Un bon go-to-market part d'abord du client, du probleme urgent, puis du canal.

**Avant de parler lancement, clarifie ceci :**
- Qui souffre le plus du probleme ?
- Quelle douleur precise veux-tu resoudre en premier ?
- Quel petit segment peux-tu atteindre cette semaine ?
- Quelle promesse simple peux-tu tester sans construire 10 fonctionnalites ?


**Actions**: quick_prompt, quick_prompt

**Expected hits**: -

**Missing expected**: avant de choisir tes canaux, whatsapp, valide d'abord ton icp

**Forbidden hits**: -

## gtm_three_phase_plan

**Module**: gtm

**Message**: Peux-tu me proposer un plan GTM simple pour mes 3 premiers mois ?

**Reply**: Voici un plan GTM en 3 phases :

**PHASE 1 : Pre-Launch (Semaines 1-4)**
- Valider l'interet avec 10-10 contacts
- Demo personnalisee de 15 min
- Feedback sur 3 fonctionnalites cles

**PHASE 2 : Launch (Semaines 5-8)**
- 30 clients payants
- Offre lancement : -30% pendant 3 mois
- Partnership avec associations

**PHASE 3 : Post-Launch (Semaines 9-16)**
- 100 clients, churn < 5%
- Programme referral : 1 mois gratuit par parrainage
- Expansion vers d'autres villes


**Actions**: quick_prompt, quick_prompt

**Expected hits**: phase 1, phase 2, phase 3

**Missing expected**: -

**Forbidden hits**: -

## market_sizing_before_validation

**Module**: market-sizing

**Message**: Je veux calculer mon TAM avant meme de valider le probleme.

**Reply**: Tu vas trop vite. Avant de calculer TAM/SAM/SOM, il faut d'abord verifier que le probleme est reel, urgent, et assez douloureux pour une cible precise.

**Ordre recommande :**
1. Valider le probleme avec des interviews
2. Definir un ICP assez etroit
3. Valider une willingness-to-pay ou une urgence claire
4. Ensuite seulement calculer TAM, SAM et SOM


**Actions**: quick_prompt, quick_prompt

**Expected hits**: tu vas trop vite, ordre recommande, valider le probleme

**Missing expected**: -

**Forbidden hits**: -

## market_sizing_top_down

**Module**: market-sizing

**Message**: Le marche africain vaut des milliards. Je veux faire un calcul top-down.

**Reply**: Attention au top-down ! Le marche 'africain' ou 'mondial' ne veut rien dire pour ton business.

**Utilise la methode BOTTOM-UP :**
1. Compte le nombre REEL de clients potentiels
2. Multiplie par le prix qu'ils paient
3. Applique des filtres progressifs (geo, budget, conscience du probleme)

**TAM** = Tous les clients potentiels x prix
**SAM** = Clients dans ta zone geographique x prix
**SOM** = Clients que tu peux toucher en 2-3 ans x prix


**Actions**: quick_prompt, quick_prompt

**Expected hits**: attention au top-down, bottom-up, tam

**Missing expected**: -

**Forbidden hits**: -

## market_sizing_bottom_up

**Module**: market-sizing

**Message**: Explique-moi la methode bottom-up pour calculer mon TAM, SAM et SOM.

**Reply**: Voici le calcul bottom-up :

**TAM** = Tous les clients potentiels au pays x prix annuel
**SAM** = Clients dans ta ville/region x prix annuel
**SOM** = 0.5-2% du SAM (scenario realiste en 2-3 ans) x prix annuel

**HYPOTHESES A VALIDER :**
- Nombre reel de clients (verifier avec sources officielles)
- Pourcentage pret a payer (verifier par interviews)
- Prix valide par willingness-to-pay interviews


**Actions**: quick_prompt, quick_prompt

**Expected hits**: calcul bottom-up, som, hypotheses a valider

**Missing expected**: -

**Forbidden hits**: -

## roi_unfounded_big_claim

**Module**: roi

**Message**: Mon client aura 500% de ROI en 6 mois.

**Reply**: Ce chiffre doit etre justifie. Sur quoi est-il base ?

**QUESTIONS A TE POSER :**
1. **Formule utilisee** : ROI = (Gain - Investissement) / Investissement
2. **Hypotheses** : Nombre de clients? Base sur combien d'interviews?
3. **Prix** : Valide par des willingness-to-pay interviews?
4. **Pourquoi 6 mois?** : Quel precedent montre ce resultat?

**RECOMMANDATION :** Commence par un ROI conservateur base sur des donnees reelles.
Un ROI de 200% base sur des donnees reelles vaut mieux qu'un ROI de 500% base sur des suppositions.


**Actions**: quick_prompt

**Expected hits**: doit etre justifie, formule, conservateur

**Missing expected**: -

**Forbidden hits**: -

## roi_grounded_framework

**Module**: roi

**Message**: Aide-moi a calculer un ROI concret pour mes clients.

**Reply**: Voici comment calculer le ROI concret pour ton client :

**HYPOTHESES DE BASE :**
- Cout de ta solution
- Temps economise par le client
- Reduction d'erreurs ou de pertes

**FORMULE :** ROI = (Gain - Cout) / Cout
**PAYBACK :** Combien de mois pour recuperer l'investissement?


**Actions**: quick_prompt

**Expected hits**: calculer le roi concret, formule, payback

**Missing expected**: -

**Forbidden hits**: -

## research_signal_extraction

**Module**: research

**Message**: Comment extraire les signaux forts de mes interviews et retours clients ?

**Reply**: Voici comment extraire les signaux de tes interviews :

**PAIN POINTS :** Cherche les phrases avec emotion forte (frustration, colere, resignation)
**WILLINGNESS TO PAY :** Cherche les chiffres concrets ('je paierais X', 'ca me coute Y')
**BUYING SIGNALS :** Cherche les conditions precises ('si ca fait X je suis interesse')
**OBJECTIONS :** Cherche les refus ('c'est cher', 'j'ai pas besoin', 'Excel me suffit')

**SIGNAUX FORTS :** Douleur quantifiee + condition d'achat precise + solutions actuelles insatisfaisantes


**Actions**: quick_prompt

**Expected hits**: extraire les signaux, pain points, buying signals

**Missing expected**: -

**Forbidden hits**: -

## research_interview_framework

**Module**: research

**Message**: Prepare-moi une structure d'interview pour valider un probleme.

**Reply**: Voici le framework pour une interview efficace :

**PREPARATION (15 min) :**
1. Objectif clair : Valider que [profil] souffre de [probleme]
2. Hypotheses a tester : Ils perdent X, utilisent Y, pret a payer Z

**STRUCTURE (30-45 min) :**
- Introduction (3 min) : 'Je ne vends rien, je cherche a comprendre'
- Contexte (5 min) : 'Parle-moi de ton activite'
- Exploration probleme (15 min) : 'Qu'est-ce qui te prend le plus de temps?'
- Validation solution (10 min) : 'Si une solution existait, qu'est-ce qu'elle devrait faire?'
- Conclusion (2 min) : 'Y a-t-il d'autres personnes qui vivent la meme chose?'


**Actions**: quick_prompt

**Expected hits**: framework pour une interview efficace, preparation, structure

**Missing expected**: -

**Forbidden hits**: -

## journey_baseline_framework

**Module**: journey

**Message**: Aide-moi a cartographier le parcours de mon client.

**Reply**: Voici le parcours complet de ton client ideal :

**PHASE 1 : AWARENESS** - Decouverte du probleme
**PHASE 2 : CONSIDERATION** - Recherche de solutions
**PHASE 3 : PURCHASE** - Decision d'achat
**PHASE 4 : RETENTION** - Utilisation quotidienne
**PHASE 5 : ADVOCACY** - Recommandation

**METRIQUES PAR PHASE :**
- Awareness : Reach, mentions
- Consideration : Taux reponse demo → inscription (objectif : 40%)
- Purchase : Conversion essai → payant (objectif : 60%)
- Retention : Churn mensuel (objectif : < 5%)
- Advocacy : NPS (objectif : > 40), referrals/mois


**Actions**: quick_prompt

**Expected hits**: awareness, consideration, retention

**Missing expected**: -

**Forbidden hits**: -