"""Configuration settings for DBpedia GraphRAG system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DBpediaConfig:
    """DBpedia SPARQL endpoint configuration."""

    sparql_endpoint: str = "https://dbpedia.org/sparql"
    spotlight_endpoint: str = "https://api.dbpedia-spotlight.org/en/annotate"
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_second: float = 10.0
    cache_ttl: int = 3600  # 1 hour


@dataclass
class RedisConfig:
    """Redis cache configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    enabled: bool = False  # Disabled by default for simpler setup


@dataclass
class TraversalConfig:
    """Graph traversal configuration."""

    max_hops: int = 3
    beam_width: int = 10
    max_nodes: int = 100
    relevance_threshold: float = 0.3


@dataclass
class LLMConfig:
    """LLM configuration."""

    provider: str = "ollama"  # ollama, openai
    model: str = "llama3.1:70b"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_base: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class ContextConfig:
    """Context assembly configuration."""

    max_tokens: int = 4000
    reserved_tokens: int = 500
    include_confidence: bool = True
    textualization_format: str = "natural"  # natural, structured, triple


@dataclass
class Config:
    """Main configuration container."""

    dbpedia: DBpediaConfig = field(default_factory=DBpediaConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    traversal: TraversalConfig = field(default_factory=TraversalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        import os

        config = cls()

        # DBpedia settings
        if endpoint := os.getenv("DBPEDIA_SPARQL_ENDPOINT"):
            config.dbpedia.sparql_endpoint = endpoint
        if spotlight := os.getenv("DBPEDIA_SPOTLIGHT_ENDPOINT"):
            config.dbpedia.spotlight_endpoint = spotlight

        # Redis settings
        if redis_host := os.getenv("REDIS_HOST"):
            config.redis.host = redis_host
            config.redis.enabled = True
        if redis_port := os.getenv("REDIS_PORT"):
            config.redis.port = int(redis_port)
        if redis_password := os.getenv("REDIS_PASSWORD"):
            config.redis.password = redis_password

        # LLM settings
        if llm_provider := os.getenv("LLM_PROVIDER"):
            config.llm.provider = llm_provider
        if llm_model := os.getenv("LLM_MODEL"):
            config.llm.model = llm_model
        if api_key := os.getenv("OPENAI_API_KEY"):
            config.llm.api_key = api_key

        return config


# Wikidata to DBpedia property mapping
# Based on 2WikiMultiHopQA benchmark properties
WIKIDATA_TO_DBPEDIA_PROPERTIES: dict[str, list[str]] = {
    # Person properties
    "P569": ["dbo:birthDate", "dbp:birthDate"],  # date of birth
    "P570": ["dbo:deathDate", "dbp:deathDate"],  # date of death
    "P19": ["dbo:birthPlace", "dbp:birthPlace"],  # place of birth
    "P20": ["dbo:deathPlace", "dbp:deathPlace"],  # place of death
    "P22": ["dbo:father", "dbp:father"],  # father
    "P25": ["dbo:mother", "dbp:mother"],  # mother
    "P26": ["dbo:spouse", "dbp:spouse"],  # spouse
    "P40": ["dbo:child", "dbp:children"],  # child
    "P3373": ["dbo:relative", "dbp:siblings"],  # sibling
    "P27": ["dbo:nationality", "dbp:nationality"],  # country of citizenship
    "P106": ["dbo:occupation", "dbp:occupation"],  # occupation
    "P69": ["dbo:almaMater", "dbp:almaMater"],  # educated at
    "P108": ["dbo:employer", "dbp:employer"],  # employer
    "P184": ["dbo:doctoralAdvisor", "dbp:doctoralAdvisor"],  # doctoral advisor
    "P185": ["dbo:doctoralStudent", "dbp:doctoralStudents"],  # doctoral student
    "P101": ["dbo:field", "dbp:field"],  # field of work
    # Film properties
    "P57": ["dbo:director", "dbp:director"],  # director
    "P162": ["dbo:producer", "dbp:producer"],  # producer
    "P577": ["dbo:releaseDate", "dbp:releaseDate"],  # publication date
    "P495": ["dbo:country", "dbp:country"],  # country of origin
    "P86": ["dbo:musicComposer", "dbp:music"],  # composer
    "P58": ["dbo:writer", "dbp:writer"],  # screenwriter
    # Organization properties
    "P112": ["dbo:foundedBy", "dbp:founder"],  # founded by
    "P571": ["dbo:foundingDate", "dbp:founded"],  # inception
    "P17": ["dbo:country", "dbp:country"],  # country
    # Album/Song properties
    "P175": ["dbo:artist", "dbp:artist"],  # performer
    "P264": ["dbo:recordLabel", "dbp:label"],  # record label
}

# Inference rules for multi-hop reasoning (from 2WikiMultiHopQA)
INFERENCE_RULES: dict[str, tuple[str, str]] = {
    "paternal_grandfather": ("father", "father"),
    "paternal_grandmother": ("father", "mother"),
    "maternal_grandfather": ("mother", "father"),
    "maternal_grandmother": ("mother", "mother"),
    "grandchild": ("child", "child"),
    "father-in-law": ("spouse", "father"),
    "mother-in-law": ("spouse", "mother"),
    "sibling-in-law": ("spouse", "sibling"),
    "child-in-law": ("child", "spouse"),
    "uncle_paternal": ("father", "sibling"),
    "uncle_maternal": ("mother", "sibling"),
    "aunt_paternal": ("father", "sibling"),
    "aunt_maternal": ("mother", "sibling"),
}

# Question type patterns for classification
QUESTION_TYPE_PATTERNS: dict[str, list[str]] = {
    "comparison": [
        r"who (was|is) born (first|earlier|later)",
        r"who (died|lived) (first|longer|earlier)",
        r"which .* (older|younger|first|earlier|later)",
        r"are both .* from the same",
        r"do both .* have the same",
        r"(more|less|greater|fewer) than",
    ],
    "inference": [
        r"who is the (grandfather|grandmother|grandchild)",
        r"who is the (father|mother|uncle|aunt)-in-law",
        r"who is the (paternal|maternal) (grandfather|grandmother)",
        r"who is .+'s (grandfather|grandmother)",
    ],
    "compositional": [
        r"who (is|was) the .* of the .* (of|who|that)",
        r"what is the .* of the .* (of|who|that)",
        r"where (is|was|did) the .* of",
        r"when (is|was|did) the .* of",
    ],
    "bridge_comparison": [
        r"which (film|movie) has the director (born|who)",
        r"which .* has the .* (born|who was born)",
        r"are both director of .* from",
        r"do both .* of .* have",
    ],
}
