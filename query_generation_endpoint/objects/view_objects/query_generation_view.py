from query_generation_endpoint.objects.query_generation.query_generation_agent import (
    query_generation_agent,
)
from query_generation_endpoint.objects.query_generation.config_retrieval_query_agent import (
    config_retrieval_query_agent,
)
from difflib import SequenceMatcher
from ACI_AI_Backend.objects.chromadb_client import chromadb_client
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import logging


logger = logging.getLogger(__name__)


class QueryGenerationView(APIView):
    """
    Handle SIEM query generation requests.

    Supports two request modes:
    - **Prompt mode** – supply ``prompt`` (optional ``siem`` and ``web_search``).
    - **Structured mode** – supply the required case/task/activity fields when ``prompt`` is omitted.

    Required structured fields:
        case_title, case_description,
        task_title, task_description, activity

    Optional fields:
        prompt (str), siem (str), web_search (bool | str)

    Responses:
        200: {"result": <generated query>, "sources": [<source URLs>]}
        400: Validation/formatting or unsupported SIEM error details
    """

    # Fields that must be present in structured mode
    REQUIRED_FIELDS = [
        "case_title",
        "case_description",
        "task_title",
        "task_description",
        "activity",
        "fields",
        "search_config_files",
    ]

    def _merge_overlapping_text(self, existing: str, incoming: str) -> str:
        """
        Unionize two texts by removing overlap between them.
        """

        if not existing:
            return incoming
        if not incoming:
            return existing

        existing = existing.strip()
        incoming = incoming.strip()

        if incoming in existing:
            return existing
        if existing in incoming:
            return incoming

        match = SequenceMatcher(None, existing, incoming).find_longest_match(
            0,
            len(existing),
            0,
            len(incoming),
        )

        if match.size > 0:
            existing_prefix = existing[: match.a]
            overlap = existing[match.a : match.a + match.size]
            existing_suffix = existing[match.a + match.size :]

            incoming_prefix = incoming[: match.b]
            incoming_suffix = incoming[match.b + match.size :]

            return f"{existing_prefix}{incoming_prefix}{overlap}{incoming_suffix}{existing_suffix}"

        return f"{existing}\n{incoming}"

    def _build_config_search_terms(self, investigation_text: str) -> list[str]:
        """Build SIEM-config-focused search terms from the investigation activity.

        Uses the keyword generation agent to distill the activity into a single
        focused retrieval query. Falls back to the raw activity text on failure.
        """

        if not investigation_text or not investigation_text.strip():
            return []

        base_text = investigation_text.strip()

        try:
            generated_query = config_retrieval_query_agent.invoke(base_text)
        except Exception:
            logger.exception("Keyword generation failed for config retrieval")
            generated_query = ""

        if generated_query and generated_query.lower() != base_text.lower():
            return [generated_query]

        return [base_text]

    def _search_relevant_config_files(self, search_terms: list[str]) -> dict[str, str]:
        """
        Return mapping from filename to unionized content from semantic matches
        across one or more keyword-augmented search terms.
        """

        if not search_terms:
            return {}

        collection = chromadb_client.get_or_create_collection("config_files")
        relevant_config_file_contents: dict[str, str] = {}

        for term in search_terms:
            query_result = collection.query(query_texts=[term], n_results=6)
            documents = query_result.get("documents", [[]])
            metadatas = query_result.get("metadatas", [[]])
            if not documents or not metadatas:
                continue

            for doc, metadata in zip(documents[0], metadatas[0]):
                if not isinstance(metadata, dict):
                    continue

                filename = metadata.get("filename")
                if not filename:
                    continue

                if filename in relevant_config_file_contents:
                    relevant_config_file_contents[filename] = self._merge_overlapping_text(
                        relevant_config_file_contents[filename],
                        doc,
                    )
                else:
                    relevant_config_file_contents[filename] = doc

        return relevant_config_file_contents

    def post(self, request, *args, **kwargs):
        data = request.data

        user_prompt = data.get("prompt")
        siem = data.get("siem")

        # -------------------------------------------------
        # Validation of structured fields (when no prompt)
        # -------------------------------------------------
        missing_fields: list[str] = []
        empty_fields: list[str] = []

        if not user_prompt:
            for field in self.REQUIRED_FIELDS:
                if field not in data:
                    missing_fields.append(field)
                elif str(data.get(field)).strip() == "":
                    empty_fields.append(field)

        if missing_fields or empty_fields:
            return Response(
                {
                    "error": "Invalid parameters",
                    "missing_fields": missing_fields,
                    "empty_fields": empty_fields,
                    "note": "Structured fields are required when 'prompt' is not provided",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # -------------------------------------------------
        # Extract common fields
        # -------------------------------------------------
        case_title = data.get("case_title")
        case_description = data.get("case_description")
        task_title = data.get("task_title")
        task_description = data.get("task_description")
        activity = data.get("activity")
        fields = data.get("fields")
        search_config_files = data.get("search_config_files").lower() == "true"
        prev_activity_critique = data.get("prev_activity_critique", None)
        max_queries_per_iteration = data.get("max_queries_per_iteration", None)
        earliest_unit = data.get("earliest_unit", None)
        earliest_magnitude = data.get("earliest_magnitude", None)
        vicinity_unit = data.get("vicinity_unit", None)
        vicinity_magnitude = data.get("vicinity_magnitude", None)
        web_search_enabled = data.get("web_search", False)
        additional_notes = data.get("additional_notes", None)

        # -------------------------------------------------
        # Normalise web_search flag
        # -------------------------------------------------
        if isinstance(web_search_enabled, str):
            lw = web_search_enabled.lower()
            if lw.isdigit():
                web_search_enabled = bool(int(lw))
            elif lw in {"true", "false"}:
                web_search_enabled = lw == "true"
            else:
                return Response(
                    {"error": 'Parameter "web_search" is incorrectly formatted'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if prev_activity_critique is not None:
            if not isinstance(prev_activity_critique, str):
                return Response(
                    {"error": 'Parameter "prev_activity_critique" must be a string'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if max_queries_per_iteration is not None:
            try:
                max_queries_per_iteration = int(max_queries_per_iteration)
            except (TypeError, ValueError):
                return Response(
                    {"error": 'Parameter "max_queries_per_iteration" must be an integer'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if max_queries_per_iteration < 1 or max_queries_per_iteration > 20:
                return Response(
                    {"error": 'Parameter "max_queries_per_iteration" must be between 1 and 20'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if additional_notes is not None and not isinstance(additional_notes, str):
            return Response(
                {"error": 'Parameter "additional_notes" must be a string or null'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # -------------------------------------------------
        # Build optional web‑search context
        # -------------------------------------------------
        context: dict | None = None
        sources: set[str] = set()

        if user_prompt and web_search_enabled:
            searcher = WebSearcher()
            context = searcher.run(user_prompt)
            for keyword in context.keys():
                sources.update(context[keyword]["sources"])

        if not user_prompt and web_search_enabled:
            # Structured query – build a composite prompt for web search
            search_query = (
                f"Case Title: {case_title}\n\n" f"Case Description: {case_description}\n\n" f"Task Title: {task_title}\n\n" f"Task Description: {task_description}\n\n" f"Activity: {activity}"
            )
            searcher = WebSearcher()
            context = searcher.run(search_query)
            for keyword in context.keys():
                sources.update(context[keyword]["sources"])

        # Searches for relevant config files chunks in vector db
        relevant_config_file_contents = None
        print("Search config files: ", search_config_files)
        if search_config_files:
            if user_prompt:
                config_search_text = user_prompt
            else:
                config_search_text = activity
            
            print("Searching config files...")
            #config_search_terms = self._build_config_search_terms(config_search_text)
            #relevant_config_file_contents = self._search_relevant_config_files(config_search_terms)

        # -------------------------------------------------
        # Invoke the query generation agent
        # -------------------------------------------------
        try:
            if user_prompt:
                query_data = query_generation_agent.invoke(
                    siem=siem,
                    user_prompt=user_prompt,
                    additional_notes=additional_notes,
                    relevant_config_file_contents=relevant_config_file_contents,
                    web_search_context=context,
                )
            else:
                query_data = query_generation_agent.invoke(
                    siem=siem,
                    case_title=case_title,
                    case_description=case_description,
                    task_title=task_title,
                    task_description=task_description,
                    activity=activity,
                    fields=fields,
                    prev_activity_critique=prev_activity_critique,
                    max_queries_per_iteration=max_queries_per_iteration,
                    earliest_unit=earliest_unit,
                    earliest_magnitude=earliest_magnitude,
                    vicinity_unit=vicinity_unit,
                    vicinity_magnitude=vicinity_magnitude,
                    additional_notes=additional_notes,
                    relevant_config_file_contents=relevant_config_file_contents,
                    web_search_context=context,
                )
        except ValueError as e:
            # Capture unsupported SIEM or argument errors
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # -------------------------------------------------
        # Final response
        # -------------------------------------------------
        if not query_data:
            # The agent returned an empty string / None → unsupported SIEM
            return Response(
                {"error": "Specified SIEM platform not implemented"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {"result": query_data, "sources": list(sources)},
            status=status.HTTP_200_OK,
        )
