import Utils

def create_html(header, formatted_div, html_fileName):
    formatted_html_filePath = 'htmlFormat\HtmlFormat.html'
    fileData = ''
    # Open the file in read mode ("r")
    with open(formatted_html_filePath, "r") as f:
        # Read the entire file contents
        fileData  = f.read()

    fileData = fileData.replace("{{headerName}}", header)
    divData = ''
    for div in formatted_div:
        divData = divData + div
        pass
    fileData = fileData.replace("{{divData}}", divData)

    html_path = f"htmlOutputDir/{html_fileName}.html"
    Utils.save_to_file(html_path, fileData)
    return html_path

