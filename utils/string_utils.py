import datetime


def get_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")[2:]

def abbreviate_number(number):
    if number < 1_000:
        return str(round(number))
    elif number < 1000_000:
        return str(round(number/1_000, 1)) + "K"
    else:
        return str(round(number/1000_000, 1)) + "M"
