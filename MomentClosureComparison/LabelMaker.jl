function LabelMaker(i)

    Vars = ["S","E","I","R"]
    mst  = "m"
    str  = "E("
    for j = 1:4
        pow = i[j]
        if pow == 0
            mst *= "₀"
        end
        if pow == 1
            str *= Vars[j]
            mst *= "₁"
        end
        if pow == 2
            str *= Vars[j] * "²"
            mst *= "₂"
        end
    end
    str *= ")"
    str = mst * " = " * str

end
