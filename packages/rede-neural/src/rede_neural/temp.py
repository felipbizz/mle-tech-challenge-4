from datetime import UTC, datetime

now = datetime.now(UTC)
print(now.strftime(format="%Y/%m/%d-%H:%M"))
