import re

with open("traffic_sign_app/ui/tabs.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_info_col = False
info_col_indent = 0

for i, line in enumerate(lines):
    if "with info_col:" in line and "def render_webcam_tab" in "".join(lines[max(0, i-200):i]):
        new_lines.append(line)
        info_col_indent = len(line) - len(line.lstrip())
        new_lines.append(" " * info_col_indent + "    @st.fragment(run_every='1.5s')\n")
        new_lines.append(" " * info_col_indent + "    def _render_info_panel():\n")
        in_info_col = True
        continue
    
    if in_info_col:
        # Check if we exited the with block (indentation <= info_col_indent)
        # But ignore empty lines
        if line.strip() == "":
            new_lines.append(line)
            continue
        
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= info_col_indent:
            in_info_col = False
            # Call the function before exiting the block? No, wait, if we exit the with info_col block, 
            # we should have called the function INSIDE the block!
            # Let's insert the function call right before this line
            new_lines.insert(-1, " " * info_col_indent + "    _render_info_panel()\n\n")
            new_lines.append(line)
            continue
        
        # Indent the line by 4 spaces
        new_lines.append("    " + line)
    else:
        new_lines.append(line)

if in_info_col:
    # If the file ends while still in the block
    new_lines.append("\n" + " " * info_col_indent + "    _render_info_panel()\n")

with open("traffic_sign_app/ui/tabs.py", "w") as f:
    f.writelines(new_lines)
print("Done")
