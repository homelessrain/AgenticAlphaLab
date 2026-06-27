from datetime import datetime, timedelta


def shift_datestr(datestr: str, days: int) -> str:
    return (datetime.strptime(datestr, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
