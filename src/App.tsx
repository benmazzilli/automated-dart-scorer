import { useMatch } from './store/match'
import { PlayScreen } from './ui/screens/PlayScreen'
import { ResultScreen } from './ui/screens/ResultScreen'
import { SetupScreen } from './ui/screens/SetupScreen'

export function App() {
  const state = useMatch((s) => s.state)

  if (!state) return <SetupScreen />
  if (state.status === 'finished') return <ResultScreen />
  return <PlayScreen />
}
