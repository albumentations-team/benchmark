"""Utility functions shared between comparison scripts."""

from pathlib import Path


def update_readme(readme_path: Path, content: str, start_marker: str, end_marker: str) -> None:
    """Update a section of the README file between markers"""
    auto_generated_comment = "<!-- This file is auto-generated. Do not edit directly. -->\n\n"

    if not readme_path.exists():
        # Create a new README if it doesn't exist
        readme_path.write_text(f"{auto_generated_comment}{start_marker}\n{content}\n{end_marker}")
        return

    # Read the existing README
    readme_content = readme_path.read_text()

    # Add auto-generated comment at the top if it doesn't exist
    if not readme_content.startswith(auto_generated_comment.strip()):
        readme_content = auto_generated_comment + readme_content

    # Find the section to update
    start_index = readme_content.find(start_marker)
    end_index = readme_content.find(end_marker)

    if start_index == -1 or end_index == -1:
        # If markers not found, append to the end
        readme_content += f"\n\n{start_marker}\n{content}\n{end_marker}"
    else:
        # Replace the section between markers
        readme_content = (
            readme_content[: start_index + len(start_marker)] + "\n" + content + "\n" + readme_content[end_index:]
        )

    # Write the updated README
    readme_path.write_text(readme_content)
