"""How old a paper is, worked out in code from its publication date and the
current date, never by the model. A pure function of its inputs: the same date
and `today` always give the same answer, and nothing is stored, so it can't
go stale in a cache or a log.

The result is deliberately coarse. OpenAlex reports YYYY-01-01 whenever only
the year is known, and nothing says when that is a placeholder, so a date
ending -01-01 is compared by year only. Anything more precise would be
false precision.
"""

import datetime


def _parse(date_text):
    try:
        return datetime.date.fromisoformat((date_text or "").strip()[:10])
    except ValueError:
        return None


def age(date_text, today=None):
    """"this year", "last year", "about 5 months ago", "about 9 years ago", and so
    on, or None if date_text isn't a date."""
    published = _parse(date_text)
    if published is None:
        return None
    today = today or datetime.date.today()
    if published > today:
        return "dated in the future (the date is unreliable)"

    if published.month == 1 and published.day == 1:  # probably only the year is known
        years = today.year - published.year
        if years == 0:
            return "this year"
        if years == 1:
            return "last year"
        return f"about {years} years ago"

    days = (today - published).days
    months = days / 30.44
    if months < 2:
        return "less than 2 months ago"
    if round(months) < 24:
        return f"about {round(months)} months ago"
    return f"about {round(days / 365.25)} years ago"


def display(date_text):
    """A date for showing to the user: the year alone when the date is probably
    only year-precise (-01-01), otherwise the full date. None if it isn't a date."""
    published = _parse(date_text)
    if published is None:
        return None
    return str(published.year) if (published.month, published.day) == (1, 1) else published.isoformat()
