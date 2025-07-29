
def print_result_as_md(results, width=10):
    
    for key, value in results.items():
        first_row = "| item        | "
        item_number = len(value) + 1
        for k, v in value.items():
            first_row += f"{k:<{width}} |"
        break

    print(first_row)
    print("|" + "----------|" * item_number)

    for key, value in results.items():
        row = f"| {key:<{width}} | "
        for k, v in value.items():
            row += f"{v:<{width}.3f} |"
        print(row)

