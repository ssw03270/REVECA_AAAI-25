import os
import glob


def count_lines_and_characters(root_folder):
    total_lines = 0
    total_chars = 0
    total_files = 0

    for filepath in glob.glob(os.path.join(root_folder, '**/*.txt'), recursive=True):
        total_files += 1
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            total_lines += len(lines)
            total_chars += sum(len(line) for line in lines)

    if total_files > 0:
        avg_lines = total_lines / 10
        avg_chars = total_chars / total_lines
    else:
        avg_lines = 0
        avg_chars = 0

    return total_lines, total_chars, avg_lines, avg_chars


root_folder = '../cwah_experiment/REVECA-llama3_Additional20_CWAH_Env/message_list/'
total_lines, total_chars, avg_lines, avg_chars = count_lines_and_characters(root_folder)

print(f"Total lines: {total_lines}")
print(f"Total characters: {total_chars}")
print(f"Average lines per file: {avg_lines}")
print(f"Average characters per file: {avg_chars}")
