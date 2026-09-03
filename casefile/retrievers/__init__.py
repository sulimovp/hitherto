from casefile.retrievers.adjacent import AdjacentProjectsRetriever
from casefile.retrievers.discourse import DiscourseRetriever
from casefile.retrievers.git_activity import GitActivityRetriever
from casefile.retrievers.github_issues import GitHubIssuesRetriever
from casefile.retrievers.github_prs import GitHubPrsRetriever
from casefile.retrievers.huggingface_discussions import HuggingFaceDiscussionsRetriever
from casefile.retrievers.process_docs import ProcessDocsRetriever
from casefile.retrievers.repo_files import RepoFilesRetriever
from casefile.retrievers.vital_signs import VitalSignsRetriever

ALL_RETRIEVERS = [
    GitHubIssuesRetriever(),
    RepoFilesRetriever(),
    GitActivityRetriever(),
    VitalSignsRetriever(),
    AdjacentProjectsRetriever(),
    GitHubPrsRetriever(),
    ProcessDocsRetriever(),
    DiscourseRetriever(),
    HuggingFaceDiscussionsRetriever(),
]
