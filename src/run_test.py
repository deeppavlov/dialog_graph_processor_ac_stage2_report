# %% [markdown]
# # First Module. Example of Use

# %% [markdown]
# The following parameters are provided as input:
# - Number of roles (currently supporting one or two roles respectively)
# - Path to the dataset in JSON format, representing a list of dialogs: *List[Dict[String, List]]*, where
# - *Dict[String, List]* has two keys and defines exactly one dialog:
#     - Key 'utterance', value - a list of dialog utterances
#     - Key 'speaker', value - a list of roles. At the i-th position, it specifies the role number of the dialog participant to whom the i-th utterance belongs.
#     - 0 - user, 1 - system
# - Dataset language
# - Number of clusters for automatic dialog graph construction

# %%
# two speakers
from DGAC_MP.intent_prediction import IntentPredictor
import pathlib
pathlib.Path("test_dir").mkdir()

DATA_PATH = "data/dataset/small_data.json"
EMBEDDINGS_FILE = "test_dir/embeddings.npy"
language = "multilingual"
num_speakers = 2
num_clusters_per_stage = [200, 30]


intent_predictor = IntentPredictor(DATA_PATH, EMBEDDINGS_FILE, language, num_speakers, num_clusters_per_stage)
intent_predictor.dialog_graph_auto_construction()
intent_predictor.dump_dialog_graph("./test_dir/dialog_graph")
intent_predictor.dgl_graphs_preprocessing()
intent_predictor.init_message_passing_model("./test_dir/")

print("Dialog Graph is complete")

# %% [markdown]
# ## Second Module. Example of Use

# %% [markdown]
# To initialize the class, the following parameters are provided:
# - Path to the binary file of the Clusters class object
# - Paths to the message passing model(s) [USER_MP_PATH, SYSTEM_MP_PATH]
# 
# For ranking:
# - Dialog: *Dict[String, List]* has two keys and defines exactly one dialog:
#     - Key 'utterance', value - a list of dialog utterances
#     - Key 'speaker', value - a list of roles. At the i-th position, it specifies the role number of the dialog participant to whom the i-th utterance belongs.
#     - 0 - user, 1 - system
# - List of utterances to be ranked
# %%
from Ranking.ranking import Ranking

# %%
# two speakers
GRAPH_PATH = "./test_dir/dialog_graph"
USER_MP_PATH = "./test_dir/GAT_system"
SYSTEM_MP_PATH = "./test_dir/GAT_user"
MODEL_PATHS = [USER_MP_PATH, SYSTEM_MP_PATH]

ranking = Ranking(GRAPH_PATH, MODEL_PATHS)


dialog = {
    'utterance': ['Wow, this is amazing!',
          "I thought you'd like it since you're interested in military history.",
          'Is there a scenic view here?',
          "Yes, there's a beautiful view of Humboldt Bay and the Samoa Peninsula.",
          'What items are featured in the exhibition?',
          'There are many items on display, including a cannon and the historic hospital building.',
          'What else can we see in the park?',
          "There's a historical garden with medicinal, edible, and decorative plants.",
          'When was the park opened?'
    ],
    'speaker': [0, 1, 0, 1, 0, 1, 0, 1, 0]
}
next_utterances = [
     'Darmstadt University of Technology is in the top 25 universities in Europe with the most scientific impact according to the QS World University Rankings 2020.',
     'In the European university rankings, Darmstadt University of Technology is ranked 15th.',
     'The maximum length of Otago Harbour is 21 kilometers (13 miles).',
     'The first event took place at the stadium on October 24, 1971.',
     'This happened 6 million years ago.',
     'The arena closed in June 2010.',
     'The park was founded in 1955.'
]
ranking_result = ranking.ranking(dialog, next_utterances)

for utterance, score in ranking_result:
    print(score, utterance)

print("Tests are passed")