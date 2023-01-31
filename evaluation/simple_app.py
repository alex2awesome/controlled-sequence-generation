## this is just for testing purposes to enable quick rendering.

from flask import Flask, render_template, request, url_for
from flask.json import jsonify
import datetime
import json, simplejson
import os, glob
import pandas as pd

app = Flask(__name__, template_folder='.')

input_data_df = pd.read_csv('data/sample_datum.csv')
if 'completed' not in input_data_df.columns:
    input_data_df['completed'] = False

output_dir = 'data/output_data'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    for f in glob.glob(os.path.join(output_dir, 'output-annotation-*.json')):
        previous_output = json.load(open(f))
        doc_id = previous_output['headline']
        if doc_id in input_data_df['headline'].unique():
            input_data_df.loc[lambda df: df['headline'] == doc_id]['completed'] = True

@app.route('/view_task', methods=['GET'])
def render_about():
    headlines = list(input_data_df['headline'].unique())
    for headline in headlines:
        if input_data_df.loc[lambda df: df['headline'] == headline]['completed'].all() == False:
            break

    # get data
    datum = input_data_df.loc[lambda df: df['headline'] == headline]
    sentences = datum['sentence_text'] #.str.replace('"', '')
    sentences = sentences.tolist()[:3]

    return render_template(
        'templates/label-sentences.html',
        sentences=sentences,
        doc_id=headline,
        do_mturk=False,
        start_time=str(datetime.datetime.now()),
    )

@app.route('/post_task', methods=['POST'])
def post_data():
    output_data = request.get_json()
    output_data['end_time'] = str(datetime.datetime.now())

    doc_id = output_data['doc_id']
    input_data_df.loc[lambda df: df['headline'] == doc_id]['completed'] = True

    ##
    output_file_idx = len(glob.glob(output_dir + '/*'))
    with open(os.path.join(output_dir, 'output-annotation-%s.json' % output_file_idx) , 'w') as f :
        json.dump(output_data, f)
    return "success"


if __name__ == '__main__':
    app.run(debug=True, port=5003)