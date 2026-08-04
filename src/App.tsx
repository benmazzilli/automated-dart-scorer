import { useState } from 'react'
import { useMatch } from './store/match'
import { PlayScreen } from './ui/screens/PlayScreen'
import { ResultScreen } from './ui/screens/ResultScreen'
import { SetupScreen } from './ui/screens/SetupScreen'
import { StatsScreen } from './ui/screens/StatsScreen'

export function App() {
  const state = useMatch((s) => s.state)
  const [showStats, setShowStats] = useState(false)

  if (state) {
    return state.status === 'finished' ? <ResultScreen /> : <PlayScreen />
  }
  if (showStats) return <StatsScreen onBack={() => setShowStats(false)} />
  return <SetupScreen onShowStats={() => setShowStats(true)} />
}
