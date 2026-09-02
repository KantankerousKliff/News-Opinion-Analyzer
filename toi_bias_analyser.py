"""
Times of India — Editorial Bias Analyzer
=========================================
Compares editorial articles across two eras:
  • Historical  : pre-2010  (GDELT archive)
  • Modern      : post-2020 (NewsAPI)

  1. SUBJECTIVITY     — ratio of subjective/emotive words via VADER +
                        Wilson et al. MPQA-style subjectivity lexicon

  2. SOURCE DIVERSITY — named-entity attribution counting via spaCy NER;
                        measures how many distinct sources are quoted

  3. AGENCY IMBALANCE — dependency-parse subject/object roles per entity
                        type (spaCy); reveals who acts vs who is acted upon

  4. MORAL FRAMING    — coverage of moral foundation categories
                        (Harm, Fairness, Authority, Loyalty, Purity)
                        via the Moral Foundations Dictionary 2.0


"""

import json
import math
import os
import re
import time
import urllib.request
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.stats as stats
import spacy
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
api_key = os.getenv("API_KEY")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", api_key)
ARTICLES_PER_ERA = 20
GDELT_RESULTS = 10  # request more; some will be empty

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
C_OLD = "#8B2500"
C_NEW = "#1A3A5C"
C_GOLD = "#C8960C"
C_BG = "#FAF6EE"
C_RULE = "#2A1F0E"
C_GRID = "#DDD5C0"

matplotlib.rcParams.update(
    {
        "figure.facecolor": C_BG,
        "axes.facecolor": C_BG,
        "axes.edgecolor": C_RULE,
        "axes.labelcolor": C_RULE,
        "xtick.color": C_RULE,
        "ytick.color": C_RULE,
        "text.color": C_RULE,
        "grid.color": C_GRID,
        "font.family": "serif",
    }
)

DIMENSIONS = [
    "Subjectivity",
    "Source Diversity\n(inverse)",
    "Agency\nImbalance",
    "Moral Framing",
]
DIMS_SHORT = [
    "Subjectivity",
    "Source\nDiversity",
    "Agency\nImbalance",
    "Moral\nFraming",
]
DIMS_KEYS = ["subjectivity", "source_diversity", "agency_imbalance", "moral_framing"]


# ══════════════════════════════════════════════
# 1.  LEXICONS
# ══════════════════════════════════════════════

# ── 1a. Subjectivity word list (MPQA-derived subset) ─────────────────────
# These ~200 seed words are drawn from the public MPQA subjectivity lexicon
# (Wilson et al. 2005).  For production use, load the full list from:
# https://mpqa.cs.pitt.edu/lexicons/subj_lexicon/
SUBJ_STRONG = {
    "outrageous",
    "appalling",
    "horrific",
    "scandalous",
    "disgraceful",
    "shameful",
    "despicable",
    "atrocious",
    "abhorrent",
    "heinous",
    "egregious",
    "vile",
    "corrupt",
    "catastrophic",
    "devastating",
    "disastrous",
    "alarming",
    "shocking",
    "startling",
    "remarkable",
    "extraordinary",
    "unprecedented",
    "dramatic",
    "radical",
    "extreme",
    "absolute",
    "complete",
    "total",
    "utter",
    "sheer",
    "mere",
    "blatant",
    "flagrant",
    "obvious",
    "clear",
    "glaring",
    "undeniable",
    "unmistakable",
    "inevitable",
    "certain",
    "wonderful",
    "magnificent",
    "glorious",
    "brilliant",
    "excellent",
    "outstanding",
    "superb",
    "splendid",
    "fantastic",
    "incredible",
    "amazing",
    "spectacular",
    "heroic",
    "cowardly",
    "treacherous",
    "ruthless",
    "brutal",
    "savage",
    "barbaric",
    "cruel",
    "sinister",
    "insidious",
    "nefarious",
    "diabolical",
    "vicious",
    "callous",
    "reckless",
    "irresponsible",
    "incompetent",
    "inept",
    "negligent",
    "derelict",
    "hypocritical",
    "opportunistic",
    "cynical",
    "manipulative",
    "deceptive",
    "arrogant",
    "pompous",
    "brazen",
    "audacious",
    "impudent",
    "insolent",
    "pathetic",
    "ridiculous",
    "absurd",
    "preposterous",
    "ludicrous",
    "farcical",
    "desperate",
    "urgent",
    "critical",
    "crucial",
    "vital",
    "imperative",
    "essential",
    "must",
    "demand",
    "insist",
    "force",
    "compel",
    "threaten",
    "warn",
    "condemn",
    "denounce",
    "attack",
    "assault",
    "destroy",
    "undermine",
    "betray",
    "abandon",
}
SUBJ_WEAK = {
    "suggest",
    "indicate",
    "appear",
    "seem",
    "possibly",
    "perhaps",
    "maybe",
    "likely",
    "probably",
    "generally",
    "typically",
    "often",
    "usually",
    "sometimes",
    "concern",
    "issue",
    "problem",
    "challenge",
    "difficulty",
    "question",
    "matter",
    "important",
    "significant",
    "major",
    "serious",
    "considerable",
    "substantial",
    "believe",
    "think",
    "feel",
    "argue",
    "claim",
    "contend",
    "assert",
    "maintain",
    "good",
    "bad",
    "better",
    "worse",
    "best",
    "worst",
    "positive",
    "negative",
    "support",
    "oppose",
    "agree",
    "disagree",
    "welcome",
    "reject",
    "accept",
    "deny",
}


def subjectivity_score(tokens: list[str]) -> float:
    """Ratio of subjective tokens to total content words (0–1).
    Weights strong subjective words 2x weak ones."""
    if not tokens:
        return 0.0
    lower = [t.lower() for t in tokens if t.isalpha() and len(t) > 2]
    if not lower:
        return 0.0
    strong = sum(1 for w in lower if w in SUBJ_STRONG)
    weak = sum(1 for w in lower if w in SUBJ_WEAK)
    score = (strong * 2 + weak) / len(lower)
    return min(score, 1.0)


# ── 1b. Moral Foundations Dictionary ─────────────────────────────────────
MFT_FALLBACK = {
    # Harm/Care
    "harm": [
        "harm",
        "hurt",
        "suffer",
        "cruel",
        "brutal",
        "pain",
        "victim",
        "trauma",
        "abuse",
        "torture",
        "kill",
        "murder",
        "death",
        "danger",
        "safe",
        "protect",
        "care",
        "kind",
        "compassion",
        "help",
        "rescue",
        "heal",
        "mercy",
        "welfare",
    ],
    # Fairness/Reciprocity
    "fairness": [
        "fair",
        "unfair",
        "justice",
        "injustice",
        "equal",
        "inequality",
        "rights",
        "bias",
        "discrimination",
        "privilege",
        "corrupt",
        "honest",
        "cheat",
        "fraud",
    ],
    # Authority/Respect
    "authority": [
        "authority",
        "obey",
        "duty",
        "law",
        "order",
        "tradition",
        "respect",
        "subvert",
        "rebel",
        "anarchy",
        "chaos",
        "hierarchy",
        "institution",
        "government",
        "official",
        "leader",
        "rule",
        "regulation",
        "discipline",
    ],
    # Loyalty/Betrayal
    "loyalty": [
        "loyal",
        "betray",
        "traitor",
        "nation",
        "patriot",
        "solidarity",
        "unity",
        "tribe",
        "enemy",
        "ally",
        "community",
        "together",
        "devotion",
        "treason",
    ],
    # Purity/Sanctity
    "purity": [
        "pure",
        "corrupt",
        "sacred",
        "holy",
        "sin",
        "disgust",
        "contaminate",
        "pollute",
        "filth",
        "degrade",
        "decency",
        "moral",
        "immoral",
        "vice",
    ],
}


def load_mft_lexicon() -> dict[str, set[str]]:
    """Load MFT 2.0 .dic file if available, else use fallback."""
    dic_path = Path("lexicons/mfd2.dic")
    if not dic_path.exists():
        return {k: set(v) for k, v in MFT_FALLBACK.items()}

    foundations = defaultdict(set)
    # .dic format: lines are either category headers (%cat) or word<tab>cats
    in_words = False
    cat_map = {}
    with open(dic_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "%":
                in_words = not in_words
                continue
            if not in_words:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    cat_map[parts[0]] = parts[1].lower()
            else:
                parts = line.split("\t")
                word = parts[0].lower().rstrip("*")
                cats = parts[1:] if len(parts) > 1 else []
                for c in cats:
                    fname = cat_map.get(c, c).split(".")[0]
                    foundations[fname].add(word)

    return (
        dict(foundations)
        if foundations
        else {k: set(v) for k, v in MFT_FALLBACK.items()}
    )


MFT_LEXICON: dict[str, set[str]] = {}  # populated at runtime


def moral_framing_score(tokens: list[str]) -> float:
    """
    Breadth × density of moral-foundation vocabulary.
    Returns 0–1: how morally saturated and how many distinct
    foundations are invoked.
    """
    if not tokens or not MFT_LEXICON:
        return 0.0
    lower = [t.lower() for t in tokens if t.isalpha()]
    total = len(lower) or 1
    hits = defaultdict(int)
    for w in lower:
        for fname, words in MFT_LEXICON.items():
            if w in words:
                hits[fname] += 1
    n_foundations = len(hits)  # breadth  (0–5+)
    density = sum(hits.values()) / total  # density  (0–1)
    breadth_norm = n_foundations / max(len(MFT_LEXICON), 1)
    # Combined: density drives intensity, breadth rewards multi-frame coverage
    return min((density * 3 + breadth_norm) / 2, 1.0)


# ── 1c. Attribution / quotation patterns ─────────────────────────────────
ATTR_PATTERNS = re.compile(
    r"\b(according to|said|says|told|stated|noted|added|argued|claimed|"
    r"asserted|confirmed|denied|declared|announced|explained|warned|"
    r"pointed out|observed|remarked|suggested|admitted|acknowledged)\b",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════
# 2.  NLP ANALYSIS  (spaCy-based)
# ══════════════════════════════════════════════

NLP: spacy.Language = None  # loaded once in main()


def analyse_article(text: str) -> dict:
    """
    Run all four bias measurements on a single article text.
    Returns a dict with keys matching DIMS_KEYS.
    """
    doc = NLP(text[:10_000])  # cap to avoid memory issues
    tokens = [t.text for t in doc if not t.is_space]

    # ── 1. Subjectivity (VADER compound + lexicon ratio) ──────────────────
    vader = SentimentIntensityAnalyzer()
    sents = list(doc.sents)
    if sents:
        # Mean absolute compound across sentences — captures strong sentiment
        # in either direction (positive editorials are also biased)
        vader_scores = [abs(vader.polarity_scores(s.text)["compound"]) for s in sents]
        vader_mean = float(np.mean(vader_scores))
    else:
        vader_mean = 0.0
    lex_subj = subjectivity_score(tokens)
    # Blend: VADER captures intensity, lexicon captures loaded word choice
    subjectivity = min((vader_mean * 0.6 + lex_subj * 0.4), 1.0)

    # ── 2. Source diversity (inverse — high = few sources) ────────────────
    # Count distinct named entities that appear near attribution verbs
    attribution_count = len(ATTR_PATTERNS.findall(text))
    named_sources: set[str] = set()
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "NORP"}:
            # Check if entity is near an attribution verb (within 10 tokens)
            window_start = max(0, ent.start - 10)
            window_end = min(len(doc), ent.end + 10)
            window_text = doc[window_start:window_end].text
            if ATTR_PATTERNS.search(window_text):
                named_sources.add(ent.text.lower())

    # Inverse: more unique sources = lower score (less one-sided)
    # Logistic-ish normalisation: 0 sources → 1.0, 5+ sources → ~0.1
    n_sources = len(named_sources)
    source_diversity_inv = math.exp(-0.5 * n_sources)  # 0–1, lower = diverse

    # ── 3. Agency imbalance ───────────────────────────────────────────────
    # For each sentence, classify named entities as subject (actor) or
    # object (acted-upon) using dependency parse.
    subject_ents: Counter = Counter()
    object_ents: Counter = Counter()

    for sent in sents:
        sent_doc = sent.as_doc()
        for token in sent_doc:
            if token.dep_ in {"nsubj", "nsubjpass", "csubj"}:
                chunk = token.text.lower()
                subject_ents[chunk] += 1
            elif token.dep_ in {"dobj", "iobj", "pobj", "nsubjpass"}:
                chunk = token.text.lower()
                object_ents[chunk] += 1

    total_roles = sum(subject_ents.values()) + sum(object_ents.values())
    if total_roles == 0:
        agency_imbalance = 0.0
    else:
        # Entities that appear overwhelmingly as objects (acted-upon)
        # without also appearing as subjects indicate framing imbalance
        overlap = set(subject_ents) & set(object_ents)
        only_object = sum(object_ents[e] for e in object_ents if e not in subject_ents)
        only_subject = sum(
            subject_ents[e] for e in subject_ents if e not in object_ents
        )
        imbalance_ratio = abs(only_subject - only_object) / (total_roles + 1)
        agency_imbalance = min(imbalance_ratio * 2, 1.0)

    # ── 4. Moral framing ──────────────────────────────────────────────────
    moral_framing = moral_framing_score(tokens)

    return {
        "subjectivity": round(subjectivity, 4),
        "source_diversity": round(source_diversity_inv, 4),  # already inverted
        "agency_imbalance": round(agency_imbalance, 4),
        "moral_framing": round(moral_framing, 4),
    }


# ══════════════════════════════════════════════
# 3.  DATA FETCHING
# ══════════════════════════════════════════════


def fetch_gdelt_articles(max_results: int = 30) -> list[dict]:
    print("📰  Fetching historical TOI articles from GDELT …")
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        "?query=source:timesofindia.indiatimes.com"
        "&mode=artlist"
        f"&maxrecords={max_results}"
        "&startdatetime=20050101000000"
        "&enddatetime=20091231235959"
        "&sort=DateDesc&format=json"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        results = []
        for a in articles:
            title = a.get("title", "").strip()
            text = _fetch_text(a.get("url", "")) or title
            if title:
                results.append(
                    {
                        "title": title,
                        "text": text,
                        "date": a.get("seendate", "pre-2010"),
                        "era": "historical",
                    }
                )
        print(f"   ✓ {len(results)} articles")
        return results
    except Exception as e:
        print(f"   ✗ GDELT failed ({e}), using stubs.")
        return []


def fetch_newsapi_articles(max_results: int = 25) -> list[dict]:
    print("📰  Fetching modern TOI articles from NewsAPI …")
    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY_HERE":
        print("   ⚠  No NewsAPI key — using stub data.")
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "sources": "the-times-of-india",
        "q": "editorial OR opinion",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        results = []
        for a in resp.json().get("articles", []):
            text = " ".join(
                filter(
                    None,
                    [
                        a.get("title", ""),
                        a.get("description", ""),
                        a.get("content", ""),
                    ],
                )
            )
            results.append(
                {
                    "title": a.get("title", ""),
                    "text": text,
                    "date": a.get("publishedAt", "post-2020"),
                    "era": "modern",
                }
            )
        print(f"   ✓ {len(results)} articles")
        return results
    except Exception as e:
        print(f"   ✗ NewsAPI failed ({e}), using stubs.")
        return []


def _fetch_text(url: str, max_chars: int = 3000) -> str:
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


# ── Stub articles (used as fallback) ─────────────────────────────────────

STUBS_HIST = [
    (
        "Gujarat aftermath",
        "The Gujarat riots have left wounds that the government must address with sincerity. The victims deserve justice and the perpetrators must face exemplary punishment. This barbaric violence shames the entire nation. Authorities have completely failed their duty to protect innocent citizens from these brutal attacks.",
    ),
    (
        "Nuclear deal sovereignty",
        "India's nuclear deal with the United States represents a pragmatic shift, though critics raise valid sovereignty concerns. The agreement, according to experts, will secure our energy future. Opposition leaders have argued the terms compromise strategic autonomy, while government officials claim it enhances India's standing.",
    ),
    (
        "Vidarbha farmer crisis",
        "The farmer suicides in Vidarbha are a national shame. Government policy has catastrophically failed the agrarian backbone of India. According to NGO reports, over 4,000 farmers have died. The state administration has callously ignored repeated warnings from agricultural scientists and civil society organisations.",
    ),
    (
        "Reservation politics",
        "The reservation debate in higher education pits merit against social justice. Proponents argue historical discrimination demands structural remedy. Critics contend the policy undermines academic standards. Sociologists suggest targeted economic support may serve both goals more effectively than caste-based quotas.",
    ),
    (
        "Naxalism security",
        "The Naxal insurgency spreads through central India like a cancer. Security forces have warned of deteriorating conditions. Tribal rights activists argue brutal poverty and state neglect fuel the rebellion. Independent analysts suggest neither pure military response nor naive dialogue will suffice.",
    ),
    (
        "IT sector inequality",
        "India's IT boom creates an urban elite while rural India is heartlessly abandoned. The grotesque disparity between Gurgaon's gleaming towers and village poverty represents a moral catastrophe. Economists note, however, that technology sector growth has generated significant downstream employment.",
    ),
    (
        "Pakistan terrorism",
        "Pakistan's duplicitous support for terrorism is an open secret that India can no longer tolerate. Foreign policy analysts say New Delhi must signal consequences clearly. Islamabad denies all allegations. Strategic affairs commentators are divided on whether diplomatic isolation or economic pressure would prove more effective.",
    ),
    (
        "2G scam judiciary",
        "The Supreme Court's intervention in the telecom scandal exposes rot at the heart of policy. Investigators allege billions were siphoned through fraudulent spectrum allocation. The accused maintain their innocence. Legal experts warn against convicting on presumption; transparency advocates demand swift accountability.",
    ),
    (
        "Communal harmony threat",
        "Communal harmony, India's greatest strength, is under relentless assault from cynical political entrepreneurs who profit from manufactured hatred. Religious leaders from multiple faiths have condemned the violence. Sociologists warn that institutional trust is eroding dangerously across communities.",
    ),
    (
        "Commonwealth Games corruption",
        "The Commonwealth Games fiasco is symptomatic of governance culture where accountability is sacrificed on political expediency. The Comptroller's report documents systematic waste. Organising committee officials contest the findings. Urban planners note some infrastructure legacy will benefit citizens long-term.",
    ),
    (
        "China LAC encroachment",
        "China's encroachments along the LAC demand a firm response. Former military chiefs warn India cannot afford complacency. The foreign ministry says diplomatic channels remain open. Strategic analysts debate whether economic interdependence constrains both sides' willingness to escalate.",
    ),
    (
        "Rural employment scheme",
        "The rural employment guarantee scheme represents a genuine attempt to alleviate poverty, even if implementation remains patchy and corruption-ridden. Economists at NCAER find significant consumption gains in participating districts. State governments report administrative challenges. Civil society groups document both successes and failures unevenly across states.",
    ),
    (
        "Media independence",
        "Corporate ownership of media poses a grave threat to editorial independence. Journalists' unions report growing self-censorship. Several editors deny any proprietorial interference. Media scholars argue structural conflicts of interest are inherent when news organisations are subsidiaries of industrial conglomerates.",
    ),
    (
        "Rising middle class",
        "India's rising middle class is a transformative force, but its aspirations must be matched by capable governance. Consumer surveys show surging confidence. Infrastructure bottlenecks, according to industry bodies, threaten to choke growth. Economists disagree on whether public investment or private deregulation better serves expansion.",
    ),
    (
        "Election violence Bihar",
        "Election-related violence in Bihar is a disgrace that electoral authorities have inexcusably failed to prevent. The Election Commission has deployed central forces. Local officials blame opposition provocateurs. Human rights monitors document incidents attributed to multiple parties across the political spectrum.",
    ),
    (
        "SEZ displacement",
        "Special Economic Zones are displacing lakhs of farmers through outright theft of ancestral land. Activist groups have documented coercive acquisition. Government officials cite legal compensation procedures. Courts are hearing multiple challenges; outcomes vary significantly by state and project type.",
    ),
    (
        "Healthcare public spending",
        "India's unconscionably low public health spending condemns millions to preventable death. The Planning Commission acknowledges the gap. Finance ministry officials cite fiscal constraints. Health economists argue preventive care investment yields returns that dwarf treatment costs.",
    ),
    (
        "Education RTI",
        "The Right to Education Act is a historic commitment to universal schooling, though critics warn of quality dilution. Enrollment data shows dramatic improvements. Teacher unions argue infrastructure remains woefully inadequate. Education researchers find outcomes vary sharply between states based on administrative capacity.",
    ),
    (
        "Urban infrastructure deficit",
        "India's collapsing urban infrastructure is a catastrophe in slow motion. Municipal engineers report critical shortfalls. City planners blame fragmented governance. World Bank analysts suggest metropolitan authority consolidation could unlock significant efficiency gains.",
    ),
    (
        "Climate adaptation",
        "India faces a devastating climate future unless radical policy transformation occurs immediately. Scientists at IIT describe alarming rainfall variability trends. Industry lobbies warn against hasty energy transition. Economists model costs of inaction versus adaptation investment across multiple scenarios.",
    ),
]

STUBS_MOD = [
    (
        "CAA citizenship concerns",
        "The CAA combined with proposed NRC represents, critics argue, a redefinition of citizenship along religious lines. Constitutional scholars have raised fundamental objections. The government insists no existing citizen is affected. Courts have admitted petitions challenging the legislation on equality grounds.",
    ),
    (
        "COVID governance failures",
        "India's COVID response exposed catastrophic governance failures. The oxygen crisis of April 2021 was a preventable tragedy, according to independent health analysts. Government spokespersons cite logistical achievements in vaccination. Excess mortality estimates suggest the official toll was severely undercounted.",
    ),
    (
        "Farmers protest historic",
        "The farmers' protest represents one of the largest sustained democratic agitations in recent history. Agricultural economists question whether the three farm laws served small farmers' interests. Government officials argued the reforms would raise incomes. Parliament repealed the legislation after prolonged protests.",
    ),
    (
        "Galwan strategic response",
        "China's aggression in Galwan demands a comprehensive strategic response beyond military posturing. Former NSA advisors urge alliance diversification. The external affairs ministry emphasises diplomatic engagement. Strategic analysts debate whether economic decoupling is feasible given deep trade integration.",
    ),
    (
        "NEP federalism",
        "The National Education Policy has bold ambitions, but its approach to medium of instruction raises federalism concerns. Linguists warn of regional language erosion. Central government officials emphasise multilingual flexibility. State governments in South India have formally protested perceived Hindi imposition.",
    ),
    (
        "Electoral bonds opacity",
        "Electoral bonds have created an opaque system of political financing that critics say institutionalises crony capitalism. The Election Commission sought disclosure requirements. The government cited donor protection rationale. The Supreme Court ultimately struck down the scheme citing constitutional violation.",
    ),
    (
        "Bulldozer extrajudicial",
        "The bulldozer has emerged as a symbol of extrajudicial punishment in several states. Legal scholars document demolitions without due process. State officials argue illegal encroachments are being removed lawfully. Human rights organisations note a pattern of targeting specific communities.",
    ),
    (
        "Startup brain drain",
        "India's startup ecosystem is a genuine success story, but the talent drain threatens to hollow out domestic innovation capacity. NASSCOM data shows record venture funding. Founders cite regulatory friction and quality-of-life factors driving migration. Policy analysts debate visa reform approaches.",
    ),
    (
        "Hijab controversy schools",
        "The hijab controversy in Karnataka schools became a flashpoint for competing claims about religious freedom and institutional uniformity. Constitutional lawyers cite Article 25 protections. State education officials argue uniform dress codes apply equally. Courts issued divided rulings before the matter reached the Supreme Court.",
    ),
    (
        "Digital surveillance capitalism",
        "Digital India's transformative potential is shadowed by surveillance capitalism and data colonialism by global technology platforms. Privacy advocates document extensive behavioural profiling. Industry representatives argue personalisation improves user experience. The Personal Data Protection Bill's final form will determine the regulatory balance.",
    ),
    (
        "Manipur crisis silence",
        "The Manipur crisis has exposed catastrophic state failure to protect citizens. International human rights bodies have demanded accountability. The central government dispatched security forces and mediators. Journalists face severe access restrictions; independent verification of events remains extremely difficult.",
    ),
    (
        "G20 diplomatic success",
        "India's G20 presidency delivered genuine diplomatic achievements, projecting soft power at a moment of global fracture. The New Delhi Declaration achieved consensus on contentious Ukraine language. Opposition parties acknowledge the diplomatic feat while questioning domestic priorities. Analysts assess the presidency's long-term strategic dividend.",
    ),
    (
        "Investigative agencies political",
        "The deployment of investigative agencies against political opponents raises serious concerns about institutional independence. Opposition leaders document a pattern of pre-election raids. Government officials cite evidence-based prosecutions. Retired bureaucrats and former judges have publicly expressed institutional concern.",
    ),
    (
        "Inequality social cohesion",
        "India's growth story is real but its benefits are grotesquely concentrated at the top. Oxfam data documents extreme wealth divergence. Government economists cite poverty reduction metrics. Sociologists warn that inequality at this scale generates social tensions visible in urban-rural and caste-based friction.",
    ),
    (
        "BBC documentary press freedom",
        "The controversy over a BBC documentary reveals governmental sensitivity to critical journalism. Press freedom indices show India's ranking declining sharply. Information ministry officials describe the documentary as propaganda. Media freedom advocates argue the response demonstrates exactly the problem the documentary raised.",
    ),
    (
        "Electoral democracy erosion",
        "India's democratic institutions face unprecedented stress, according to political scientists across the ideological spectrum. Institutional quality indices show declining scores. Government spokespersons cite electoral participation records as proof of democratic health. Constitutional scholars distinguish procedural democracy from liberal democracy.",
    ),
    (
        "Adani controversy governance",
        "Questions surrounding the Adani group have raised corporate governance concerns that regulators cannot afford to ignore. Short-seller reports alleged financial irregularities. Group spokespeople issued comprehensive rebuttals. SEBI investigations and parliamentary scrutiny committees continue examining the evidence.",
    ),
    (
        "Urban heat island crisis",
        "India's cities are becoming dangerously uninhabitable due to the urban heat island effect compounded by climate change. Heat action plans exist in few cities. Municipal officials cite resource constraints. Public health researchers document excess mortality from heat stress concentrated among outdoor workers and elderly populations.",
    ),
    (
        "Judicial appointments collegium",
        "The collegium system of judicial appointments remains opaque and resistant to accountability, critics contend. The Law Commission has repeatedly recommended reforms. The judiciary argues executive involvement threatens independence. Constitutional scholars find merit in both positions and propose hybrid models.",
    ),
    (
        "Gig economy worker rights",
        "Platform workers occupy a legal grey zone that exposes them to exploitation without recourse. Labour ministry consultation papers acknowledge the regulatory gap. Gig platforms argue flexibility is valued by workers. Union organisers document cases of arbitrary deactivation and wage theft without grievance mechanisms.",
    ),
]


def _stubs(era: str, n: int) -> list[dict]:
    pool = STUBS_HIST if era == "historical" else STUBS_MOD
    return [
        {
            "title": t,
            "text": body,
            "era": era,
            "date": f"200{i % 7 + 2}-01-01"
            if era == "historical"
            else f"202{i % 4 + 0}-01-01",
        }
        for i, (t, body) in enumerate(pool[:n])
    ]


# ══════════════════════════════════════════════
# 4.  SCORING PIPELINE
# ══════════════════════════════════════════════


def score_all(articles: list[dict]) -> list[dict]:
    scored = []
    for i, art in enumerate(articles):
        label = "HIST" if art["era"] == "historical" else " MOD"
        print(f"   [{label}] {i + 1:>2}/{len(articles)}  {art['title'][:55]}")
        scores = analyse_article(art["text"])
        art.update(scores)
        scored.append(art)
    return scored


# ══════════════════════════════════════════════
# 5.  PLOTTING
# ══════════════════════════════════════════════


def _split(articles):
    h = [a for a in articles if a["era"] == "historical"]
    m = [a for a in articles if a["era"] == "modern"]
    return h, m


def _vals(group, key):
    return np.array([a[key] for a in group], dtype=float)


def _mean_sem(group, key):
    v = _vals(group, key)
    return np.mean(v), (np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


# ── Plot 1: Grouped bar + radar ───────────────────────────────────────────


def plot_main(articles, out="toi_bias_comparison.png"):
    hist, mod = _split(articles)
    fig = plt.figure(figsize=(15, 6), facecolor=C_BG)
    fig.suptitle(
        "Times of India — Editorial Bias: Pre-2010 vs Post-2020",
        fontsize=14,
        fontweight="bold",
        color=C_RULE,
        y=1.01,
    )
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1], wspace=0.38)

    # grouped bar
    ax1 = fig.add_subplot(gs[0])
    x, w = np.arange(4), 0.32
    means_h = [_mean_sem(hist, k)[0] for k in DIMS_KEYS]
    sems_h = [_mean_sem(hist, k)[1] for k in DIMS_KEYS]
    means_m = [_mean_sem(mod, k)[0] for k in DIMS_KEYS]
    sems_m = [_mean_sem(mod, k)[1] for k in DIMS_KEYS]

    b1 = ax1.bar(
        x - w / 2,
        means_h,
        w,
        yerr=sems_h,
        color=C_OLD,
        alpha=0.85,
        capsize=4,
        label=f"Historical  n={len(hist)}",
        error_kw={"elinewidth": 1.2},
    )
    b2 = ax1.bar(
        x + w / 2,
        means_m,
        w,
        yerr=sems_m,
        color=C_NEW,
        alpha=0.85,
        capsize=4,
        label=f"Modern      n={len(mod)}",
        error_kw={"elinewidth": 1.2},
    )

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.008,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=C_RULE,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(DIMS_SHORT, fontsize=9.5)
    ax1.set_ylabel("Score  (0 = neutral,  1 = extreme)", fontsize=9)
    ax1.set_ylim(0, 1.12)
    ax1.set_title("Mean Bias per Dimension  (±1 SEM)", fontsize=10, pad=8)
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.55)
    ax1.set_axisbelow(True)

    # significance annotations
    for i, k in enumerate(DIMS_KEYS):
        h_v = _vals(hist, k)
        m_v = _vals(mod, k)
        if len(h_v) > 1 and len(m_v) > 1:
            _, p = stats.ttest_ind(h_v, m_v, equal_var=False)
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            y_top = max(means_h[i] + sems_h[i], means_m[i] + sems_m[i]) + 0.04
            ax1.text(
                i, y_top, sig, ha="center", fontsize=9, color=C_GOLD, fontweight="bold"
            )

    # radar
    ax2 = fig.add_subplot(gs[1], polar=True)
    N = 4
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    v_h = means_h + [means_h[0]]
    v_m = means_m + [means_m[0]]

    ax2.plot(angles, v_h, color=C_OLD, lw=2)
    ax2.fill(angles, v_h, color=C_OLD, alpha=0.20)
    ax2.plot(angles, v_m, color=C_NEW, lw=2)
    ax2.fill(angles, v_m, color=C_NEW, alpha=0.20)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(
        ["Subjectivity", "Source\nDiversity", "Agency\nImbalance", "Moral\nFraming"],
        fontsize=8.5,
    )
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6.5)
    ax2.set_facecolor(C_BG)
    ax2.yaxis.grid(True, linestyle="--", color=C_GRID)
    ax2.xaxis.grid(True, linestyle="--", color=C_GRID)
    ax2.set_title("Bias Profile", fontsize=10, pad=16)
    handles = [
        mpatches.Patch(color=C_OLD, alpha=0.7, label="Historical"),
        mpatches.Patch(color=C_NEW, alpha=0.7, label="Modern"),
    ]
    ax2.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=8
    )

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"   ✓ {out}")


# ── Plot 2: Violin + box distributions ───────────────────────────────────


def plot_distributions(articles, out="toi_bias_distributions.png"):
    hist, mod = _split(articles)
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.5), facecolor=C_BG)
    fig.suptitle(
        "Score Distributions per Dimension  ·  Times of India",
        fontsize=13,
        fontweight="bold",
        color=C_RULE,
    )

    for ax, key, label in zip(axes, DIMS_KEYS, DIMENSIONS):
        h_v = list(_vals(hist, key))
        m_v = list(_vals(mod, key))
        if len(h_v) > 1 and len(m_v) > 1:
            parts = ax.violinplot(
                [h_v, m_v],
                positions=[1, 2],
                widths=0.55,
                showmedians=False,
                showextrema=False,
            )
            for pc, col in zip(parts["bodies"], [C_OLD, C_NEW]):
                pc.set_facecolor(col)
                pc.set_alpha(0.40)
        bp = ax.boxplot(
            [h_v, m_v],
            positions=[1, 2],
            widths=0.20,
            patch_artist=True,
            medianprops=dict(color=C_GOLD, linewidth=2.2),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            flierprops=dict(marker="o", markersize=3.5, alpha=0.45),
        )
        for patch, col in zip(bp["boxes"], [C_OLD, C_NEW]):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pre-2010", "Post-2020"], fontsize=9.5)
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("Score (0–1)" if ax == axes[0] else "", fontsize=9)
        ax.set_title(label, fontsize=10, pad=7)
        ax.yaxis.grid(True, linestyle=":", alpha=0.55)
        ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=C_OLD, alpha=0.7, label="Historical (pre-2010)"),
        mpatches.Patch(color=C_NEW, alpha=0.7, label="Modern (post-2020)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"   ✓ {out}")


# ── Plot 3: Per-article scatter with delta arrows ─────────────────────────


def plot_scatter(articles, out="toi_bias_scatter.png"):
    hist, mod = _split(articles)
    fig, axes = plt.subplots(1, 4, figsize=(17, 5), facecolor=C_BG)
    fig.suptitle(
        "Per-Article Scores with Era Means  ·  Times of India",
        fontsize=13,
        fontweight="bold",
        color=C_RULE,
    )

    rng = np.random.default_rng(7)
    for ax, key, label in zip(axes, DIMS_KEYS, DIMENSIONS):
        h_v = _vals(hist, key)
        m_v = _vals(mod, key)
        jh = rng.uniform(-0.13, 0.13, len(h_v))
        jm = rng.uniform(-0.13, 0.13, len(m_v))
        ax.scatter(jh, h_v, color=C_OLD, alpha=0.70, s=48, zorder=3)
        ax.scatter(1 + jm, m_v, color=C_NEW, alpha=0.70, s=48, zorder=3)
        mh, mm = np.mean(h_v), np.mean(m_v)
        ax.hlines(mh, -0.25, 0.25, colors=C_OLD, lw=2.5, linestyle="--", zorder=4)
        ax.hlines(mm, 0.75, 1.25, colors=C_NEW, lw=2.5, linestyle="--", zorder=4)
        delt = mm - mh
        col_arrow = "#B03A2E" if delt > 0 else "#1D6A39"
        ax.annotate(
            f"Δ {delt:+.3f}",
            xy=(0.5, mh),
            xytext=(0.5, mm),
            arrowprops=dict(arrowstyle="->", color=col_arrow, lw=1.8),
            ha="center",
            fontsize=8.5,
            color=col_arrow,
            fontweight="bold",
        )
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre-2010", "Post-2020"], fontsize=9.5)
        ax.set_ylabel("Score (0–1)" if ax == axes[0] else "", fontsize=9)
        ax.set_title(label, fontsize=10, pad=7)
        ax.yaxis.grid(True, linestyle=":", alpha=0.55)
        ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"   ✓ {out}")


# ── Plot 4: Moral foundations breakdown ──────────────────────────────────


def plot_moral_foundations(articles, out="toi_moral_foundations.png"):
    """Bar chart showing which moral foundations are most invoked, per era."""
    hist, mod = _split(articles)
    foundations = list(MFT_LEXICON.keys())
    if not foundations:
        return

    def foundation_counts(group):
        totals = defaultdict(int)
        total_words = 0
        for art in group:
            tokens = [t.lower() for t in art["text"].split() if t.isalpha()]
            total_words += len(tokens) or 1
            for w in tokens:
                for fname, words in MFT_LEXICON.items():
                    if w in words:
                        totals[fname] += 1
        # Normalise per 1000 words
        n = total_words / 1000
        return {k: v / n for k, v in totals.items()}

    h_counts = foundation_counts(hist)
    m_counts = foundation_counts(mod)
    fnames = sorted(set(h_counts) | set(m_counts))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=C_BG)
    x = np.arange(len(fnames))
    w = 0.35
    ax.bar(
        x - w / 2,
        [h_counts.get(f, 0) for f in fnames],
        w,
        color=C_OLD,
        alpha=0.85,
        label="Historical (pre-2010)",
    )
    ax.bar(
        x + w / 2,
        [m_counts.get(f, 0) for f in fnames],
        w,
        color=C_NEW,
        alpha=0.85,
        label="Modern (post-2020)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in fnames], fontsize=10)
    ax.set_ylabel("Hits per 1,000 words", fontsize=9)
    ax.set_title(
        "Moral Foundations Vocabulary Frequency  ·  Times of India",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=9.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.55)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"   ✓ {out}")


# ══════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════


def main():
    global NLP, MFT_LEXICON

    print("\n╔═══════════════════════════════════════════════╗")
    print("║  Times of India — Editorial Bias Analyser      ║")
    print("║  (NLP-only: VADER · spaCy · MFT · Subj. lex.) ║")
    print("╚═══════════════════════════════════════════════╝\n")

    print("🔧  Loading NLP models …")
    NLP = spacy.load("en_core_web_sm")
    MFT_LEXICON = load_mft_lexicon()
    print(f"   ✓ spaCy loaded  |  MFT foundations: {list(MFT_LEXICON.keys())}\n")

    # Fetch
    hist_raw = fetch_gdelt_articles(GDELT_RESULTS)
    if not hist_raw:
        hist_raw = _stubs("historical", ARTICLES_PER_ERA)

    mod_raw = fetch_newsapi_articles(ARTICLES_PER_ERA + 5)
    if not mod_raw:
        mod_raw = _stubs("modern", ARTICLES_PER_ERA)

    hist_raw = hist_raw[:ARTICLES_PER_ERA]
    mod_raw = mod_raw[:ARTICLES_PER_ERA]
    all_articles = hist_raw + mod_raw

    print(f"\n📊  Analysing {len(all_articles)} articles …\n")
    scored = score_all(all_articles)

    hist_s = [a for a in scored if a["era"] == "historical"]
    mod_s = [a for a in scored if a["era"] == "modern"]
    print(f"\n   ✓ Done: {len(hist_s)} historical, {len(mod_s)} modern\n")

    if len(hist_s) < 2 or len(mod_s) < 2:
        print("   ✗ Too few articles to plot. Check your API keys.")
        return

    print("🖼   Generating charts …")
    plot_main(scored)
    plot_distributions(scored)
    plot_scatter(scored)
    plot_moral_foundations(scored)

    # Summary table
    print("\n" + "─" * 65)
    print(f"  {'Dimension':<22} {'Pre-2010':>10} {'Post-2020':>10} {'Δ':>8}  {'p':>8}")
    print("─" * 65)
    for key, label in zip(
        DIMS_KEYS, ["Subjectivity", "Src Diversity", "Agency Imbal.", "Moral Framing"]
    ):
        h_v = _vals(hist_s, key)
        m_v = _vals(mod_s, key)
        mh, mm = np.mean(h_v), np.mean(m_v)
        _, p = (
            stats.ttest_ind(h_v, m_v, equal_var=False)
            if len(h_v) > 1 and len(m_v) > 1
            else (0, 1)
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {label:<22} {mh:>10.4f} {mm:>10.4f} {mm - mh:>+8.4f}  {sig:>8}")
    print("─" * 65)
    print("\n✅  Four PNG charts saved to the current directory.\n")
    print("NOTE: If running on stub data, set NEWSAPI_KEY env var for")
    print("      live modern articles, and ensure GDELT is reachable for")
    print("      historical ones. Run setup_data.py once to fetch the MFT\n")
    print("      lexicon for richer moral foundations scoring.")


if __name__ == "__main__":
    main()
