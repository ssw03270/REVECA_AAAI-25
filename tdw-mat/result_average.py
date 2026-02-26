import json
import statistics
import os

test_env = json.load(open(os.path.join("tdw_mat/dataset/dataset_test/", "test_env.json"), "r"))

print(os.getcwd())
# Load the JSON file
file_path = "tdw_mat/results/LMs-gpt-4o-mini/3/eval_result.json"
print(os.path.exists('tdw_mat'))
with open(file_path, "r") as f:
    data = json.load(f)

# Extract results
episode_results = data["episode_results"]

stuff = []
food = []
for key in episode_results:
    if test_env[int(key)]['task'] == 'stuff':
        stuff.append(episode_results[key]['finish'])
    else:
        food.append(episode_results[key]['finish'])

# Collect values for averaging
finish_vals = [v["finish"] for v in episode_results.values()]
total_vals = [v["total"] for v in episode_results.values()]
token_costs = [v["token cost"] for v in episode_results.values()]
message_costs = [v["message cost"] for v in episode_results.values()]


# Compute averages
averages = {
    "finish": statistics.mean(finish_vals),
    "total": statistics.mean(total_vals),
    "token cost": round(statistics.mean(token_costs),2),
    "message cost": round(statistics.mean(message_costs),2),
}
print(averages)
print("food:",sum(food)/len(food))
print("stuff:",sum(stuff)/len(stuff))
