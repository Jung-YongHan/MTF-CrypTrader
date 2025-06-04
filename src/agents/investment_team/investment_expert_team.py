from src.agents.higher_team.higher_analysis_team import HigherReport
from src.agents.investment_team.order_tactician import OrderTactician
from src.agents.lower_team.lower_analysis_team import LowerReport


class InvestmentExpertTeam:
    # TODO RoundRobinTeam을 활용하여, 투자 전문가 간에 토론을 유도하도록 구현
    def __init__(self):
        self.order_tactician = OrderTactician()

    async def determine_order(
        self, higher_report: HigherReport, lower_report: LowerReport
    ):
        pass
