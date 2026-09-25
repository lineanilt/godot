import os
import re

# The folders you want to scan (adjust if you modified other folders)
TARGET_DIRS = ['drivers/gles3', 'scene/3d', 'scene/main', 'servers/visual']
EXTENSIONS = ('.cpp', '.h', '.glsl')

# Regex to capture C-style comments:
# Group 1: // single line comments
# Group 2: /* multi line comments */
COMMENT_PATTERN = re.compile(r'(//.*?$)|(/\*.*?\*/)', re.MULTILINE | re.DOTALL)


def get_comments(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    comments = []
    # Find all matches in the file
    for match in COMMENT_PATTERN.finditer(content):
        comment_text = match.group(0).strip()
        # Calculate which line number this comment starts on
        line_num = content.count('\n', 0, match.start()) + 1
        comments.append((line_num, comment_text))

    return comments


def main():
    total_comments = 0
    output_file = "all_comments_found.txt"

    with open(output_file, 'w', encoding='utf-8') as out:
        for d in TARGET_DIRS:
            if not os.path.exists(d):
                continue

            for root, _, files in os.walk(d):
                for file in files:
                    if file.endswith(EXTENSIONS):
                        filepath = os.path.join(root, file)
                        comments = get_comments(filepath)

                        if comments:
                            out.write(f"\n{'='*60}\n")
                            out.write(
                                f"FILE: {filepath} ({len(comments)} comments)\n")
                            out.write(f"{'='*60}\n")

                            for line_num, text in comments:
                                # Clean up formatting for multi-line comments in the output log
                                safe_text = text.replace('\n', '\n\t\t')
                                out.write(f"Line {line_num:<5} | {
                                          safe_text}\n")

                            total_comments += len(comments)

    print(f"Done! Found {total_comments} comments.")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
