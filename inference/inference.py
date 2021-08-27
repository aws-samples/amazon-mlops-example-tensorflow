import json
import os
from etl import fitandtokenize
from etl import cleanup

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    
    if context.request_content_type == 'application/json':
        ip = json.loads(data.read().decode('utf-8'))
        ip_mod = cleanup(list(ip))
        d = json.dumps({"instances": fitandtokenize(ip_mod).tolist()}) 
        return d if len(ip) else ''

    if context.request_content_type == 'text/csv':
        d = json.dumps({
            'instances': [fitandtokenize(x) for x in data.read().decode('utf-8').split(',')]
        })
        return d 

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        "whatever" or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type