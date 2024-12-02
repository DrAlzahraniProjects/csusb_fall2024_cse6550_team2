from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.llm.providers import register_llm_provider
from langchain_mistralai import ChatMistralAI

register_llm_provider("mistral", ChatMistralAI)

# Load the configuration from the config folder
config_path = "./config/config.yml"
config = RailsConfig.from_path(config_path)

guard_rails = RunnableRails(config=config)
