import Utils
from xhtml2pdf import pisa

def create_html(header, formatted_div, html_fileName):
    formatted_html_filePath = 'htmlFormat\\HtmlFormat.html'
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

    html_path = f"htmlOutputDir\\{html_fileName}.html"
    Utils.save_to_file(html_path, fileData)
    return html_path

def create_pdf(html_path, pdf_fileName):
    # Open the HTML file
    with open(html_path, 'rb') as f:
        html = f.read()

    # PDF options (optional)
    options = {
        'page-size': 'A4',  # Set page size
        'margin-left': '2cm',  # Set left margin
        'margin-right': '2cm',  # Set right margin
    }
    
    # Open file to write
    result_file = open(f'{pdf_fileName}.pdf', "w+b") # w+b to write in binary mode.

    # convert HTML to PDF
    pisa_status = pisa.CreatePDF(
            html,                   # the HTML to convert
            dest=result_file           # file handle to recieve result
    )

    # close output file
    result_file.close()

    result = pisa_status.err

    if not result:
        print("Successfully created PDF")
    else:
        print("Error: unable to create the PDF")    

    # return False on success and True on errors
    return result
