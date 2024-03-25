To set up authentication for SQLPad:

Create a file called '.env' in the folder Jupyter Lab regards as your root directory. If you're running in the browser, that would be the folder where your notebook is located, and if you're running in PyCharm it may be the root directory of the folder. (Run os.getcwd() in the notebook if you're not sure.) The contents of the file should be:

SQLPAD_USERNAME=<your hydraDX email address>
PASSWORD=<your SQLPad password>

No spaces, no quotes. SQLPAD_USERNAME as opposed to just USERNAME is necessary because USERNAME is already taken by Windows, and you can't change it.

With those variables set in that file, you should be good to go.